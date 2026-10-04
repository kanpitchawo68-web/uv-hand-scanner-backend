import cv2
import numpy as np
import base64
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# อนุญาตการเชื่อมต่อ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# 1. โหลดโมเดล YOLO (.pt)
# -----------------------------------------------------------

MODEL_PATH = "best.pt" 
print(f"กำลังโหลดโมเดลจาก {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("โมเดลพร้อมใช้งาน!")

@app.get("/")
async def root():
    return {"status": "online", "message": "UV Hand Scanner API is running"}

# -----------------------------------------------------------
# 2. Route ประมวลผลภาพ (แปลงเป็น Grayscale และวัดความสว่าง)
# -----------------------------------------------------------

@app.post("/scan")
async def process_frame(image_data: str = Form(...), mode: str = Form("palm")):
    try:
        # ถอดรหัส Base64 เป็นภาพ
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape

        # รัน YOLO ตรวจจับมือ
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        total_hand_pixels = 0
        total_gel_pixels = 0
        hand_count = 0

        # Mask สำหรับเก็บพื้นที่รูปทรงมือ
        combined_hand_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            hand_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # --- ขยายกรอบเผื่อนิ้วโป้งและขอบมือที่อาจจะล้น (Padding) ---
            padding = 40  # สามารถเพิ่ม/ลด ตัวเลขนี้ได้ (เช่น 30 หรือ 50)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            roi = frame[y1:y2, x1:x2]

            # ตัดขอบตามรูปทรงนิ้วมือ (แก้ปัญหาเงาสะท้อนพื้นโต๊ะติดมากับมือ)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            # 1. ขยับ Threshold ขึ้นเป็น 45 เพื่อหนีแสงสะท้อนจางๆ ที่พื้น
            _, roi_binary = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

            # 2. ตัดสะพานเชื่อมระหว่างมือกับเงาสะท้อน (MORPH_OPEN) โดยใช้ขนาด 7x7 เพื่อกัดเซาะให้ขาดจากกัน
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_open)

            # 3. ถมรูโหว่เล็กๆ ภายในมือ (MORPH_CLOSE)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel_close)
            contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # สร้าง Mask พอดีมือ
                roi_hand_mask = np.zeros_like(gray_roi)
                cv2.drawContours(roi_hand_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                combined_hand_mask[y1:y2, x1:x2] = cv2.bitwise_or(combined_hand_mask[y1:y2, x1:x2], roi_hand_mask)

                # วาดเส้นขอบมือ
                contour_offset = largest_contour + np.array([x1, y1])
                cv2.drawContours(frame, [contour_offset], -1, (0, 255, 120), 2)

                conf = float(box.conf[0])
                cv2.putText(frame, f"Hand ({int(conf*100)}%)", (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

        # -------------------------------------------------------------
        # 3. วิเคราะห์ความสว่าง (Grayscale)
        # -------------------------------------------------------------
        if hand_count > 0:
            # 3.1 แปลงภาพเต็มเป็นขาวดำ (Grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 3.2 ตั้งค่าเกณฑ์ความสว่าง Threshold (แยกตามโหมด)
            if mode == 'palm':
                brightness_threshold = 120 # หน้ามือมักจะโดนแสง UV สว่าง ชัดเจน
            else:
                brightness_threshold = 85  # หลังมือมักจะมืดกว่า ให้ลดเกณฑ์ลงเพื่อให้จับเจลติดง่ายขึ้น

            # ดึงเฉพาะจุดที่สว่างกว่า Threshold ที่ตั้งไว้
            _, bright_mask = cv2.threshold(gray_frame, brightness_threshold, 255, cv2.THRESH_BINARY)

            # 3.3 เอาเฉพาะจุดที่สว่าง "และ" อยู่บนพื้นที่มือเท่านั้น
            gel_on_hand = cv2.bitwise_and(bright_mask, bright_mask, mask=combined_hand_mask)

            # นับพิกเซล
            total_hand_pixels = cv2.countNonZero(combined_hand_mask)
            total_gel_pixels = cv2.countNonZero(gel_on_hand)

            # 3.4 ย้อมสีส่วนที่สว่างให้เป็นสีฟ้าเรืองแสง (Cyan) ให้ผู้ใช้เห็นชัดๆ
            frame[gel_on_hand > 0] = [255, 230, 0]

            # 3.5 ดรอปความสว่างของฉากหลังรอบนอกมือ
            background_mask = cv2.bitwise_not(combined_hand_mask)
            frame[background_mask > 0] = (frame[background_mask > 0] * 0.25).astype(np.uint8)



        # คำนวณเปอร์เซ็นต์
        coverage = 0.0
        if total_hand_pixels > 0:
            # ใช้สัดส่วนพื้นที่สว่างต่อพื้นที่มือ x 100
            coverage = min(100.0, (total_gel_pixels / total_hand_pixels) * 100)

        # ส่งภาพประมวลผลกลับเป็น Base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "coverage": round(coverage, 1),
            "hand_count": hand_count,
            "processed_image": f"data:image/jpeg;base64,{processed_base64}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)