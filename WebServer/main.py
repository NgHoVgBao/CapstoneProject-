import os
import json
import sqlite3
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import base64
import uuid
import cv2
import numpy as np

from fastapi import FastAPI, Request, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from insightface.app import FaceAnalysis

app = FastAPI(title="ESP32-S3 Face Recognition Hub")

# --- KHỞI TẠO AI INSIGHTFACE ---
print("-> Đang tải mô hình AI InsightFace...")
ai_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
ai_app.prepare(ctx_id=0, det_size=(640, 640))
print("-> Mô hình AI đã sẵn sàng!")

# --- THƯ MỤC LƯU TRỮ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- CƠ SỞ DỮ LIỆU SQLITE ---
DB_NAME = os.path.join(BASE_DIR, "database.db")
FRAME_WIDTH = 240
FRAME_HEIGHT = 240
RGB565_FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 2

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Bảng Lịch sử Điểm danh (Tạo mới nếu chưa có)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id_num TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT DEFAULT '',
            position TEXT NOT NULL,
            access_date TEXT NOT NULL,
            accuracy REAL DEFAULT 0.0,
            similar_to TEXT DEFAULT ''
        )
    ''')
    
    # 2. Bảng Hàng chờ Lưu trữ
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS storage_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            upload_time TEXT NOT NULL
        )
    ''')

    # 3. Migration tự động: Thêm cột 'email' nếu file DB cũ đã tồn tại bảng nhưng chưa có cột này
    cursor.execute("PRAGMA table_info(access_logs)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'email' not in columns:
        cursor.execute("ALTER TABLE access_logs ADD COLUMN email TEXT DEFAULT ''")
    if 'accuracy' not in columns:
        cursor.execute("ALTER TABLE access_logs ADD COLUMN accuracy REAL DEFAULT 0.0")
    if 'similar_to' not in columns:
        cursor.execute("ALTER TABLE access_logs ADD COLUMN similar_to TEXT DEFAULT ''")

    # Gom toàn bộ thay đổi và chỉ commit / close đúng 1 lần ở cuối
    conn.commit()
    conn.close()

init_db()

# --- HÀM XỬ LÝ AI ---
def extract_embedding_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    faces = ai_app.get(img)
    if len(faces) == 0:
        return None, img
    return faces[0].embedding, img

def decode_rgb565_frame(frame_bytes: bytes):
    """Convert raw RGB565 frame from ESP32 to OpenCV BGR."""
    if len(frame_bytes) != RGB565_FRAME_BYTES:
        return None
    pixels = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 2))
    pixels_swapped = pixels[:, :, ::-1]
    return cv2.cvtColor(pixels_swapped, cv2.COLOR_BGR5652BGR)

def frame_to_jpeg(frame_bytes: bytes):
    img = decode_rgb565_frame(frame_bytes)
    if img is None:
        return None, None

    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None, None
    return encoded.tobytes(), img

def decode_uploaded_image(image_bytes: bytes):
    if len(image_bytes) == RGB565_FRAME_BYTES:
        return frame_to_jpeg(image_bytes)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return (encoded.tobytes(), img) if ok else (None, None)

def image_data_url(jpeg_bytes: bytes):
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- HÀM DÙNG CHUNG: SO KHỚP KHUÔN MẶT + GHI LOG (dùng cho cả WebSocket lẫn REST) ---
def match_face_and_log(input_vec) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id_num, user_name, email, position, image_path, embedding FROM storage_faces WHERE embedding != '[]'")
    rows = cursor.fetchall()

    best_match = None
    max_similarity = -1.0
    THRESHOLD = 0.4

    for row in rows:
        stored_vec = np.array(json.loads(row["embedding"]))
        sim = compute_cosine_similarity(input_vec, stored_vec)
        if sim > max_similarity:
            max_similarity = sim
            most_similar_name = row["user_name"]
            if sim >= THRESHOLD:
                best_match = {
                    "user_id_num": row["user_id_num"],
                    "name": row["user_name"],
                    "email": row["email"],
                    "position": row["position"],
                    "similarity": round(float(sim) * 100, 1)
                }

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if best_match:
        cursor.execute(
            """INSERT INTO access_logs 
               (user_id_num, full_name, email, position, access_date, accuracy, similar_to) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (best_match["user_id_num"], best_match["name"], best_match["email"], 
             best_match["position"], now_str, best_match["similarity"], "")
        )
        conn.commit()
        conn.close()
        # CHỈ TRẢ VỀ ID NUMBER KHI NHẬN DIỆN THÀNH CÔNG
        return {
            "status": "success",
            "is_matched": True,
            "user_id_num": str(best_match["user_id_num"])
        }
    else:
        highest_sim = round(float(max_similarity) * 100, 1) if max_similarity > 0 else 0.0
        similar_info = f"{most_similar_name} ({highest_sim}%)" if max_similarity > 0 else "None"
        
        cursor.execute(
            """INSERT INTO access_logs 
               (user_id_num, full_name, email, position, access_date, accuracy, similar_to) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("UNKNOWN", "Stranger", "", "Stranger", now_str, highest_sim, similar_info)
        )
        conn.commit()
        conn.close()
        # KHÔNG KHỚP THÌ TRẢ VỀ UNKNOWN
        return {
            "status": "success",
            "is_matched": False,
            "user_id_num": "UNKNOWN"
        }
# --- QUẢN LÝ WEBSOCKET CLIENTS (CHO GIAO DIỆN WEB MONITOR) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


# =====================================================================
# 🔌 WEBSOCKET ENDPOINTS
# =====================================================================

async def process_register_frame(frame_bytes: bytes, websocket: WebSocket):
    jpeg_bytes, img = frame_to_jpeg(frame_bytes)
    if jpeg_bytes is None:
        return  # Chỉ return ngắt luồng, KHÔNG gửi JSON về ESP32 nữa

    timestamp = int(time.time())
    file_name = f"ESP32_CAP_{timestamp}.jpg"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    cv2.imwrite(file_path, img)

    web_image_path = f"/uploads/{file_name}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO storage_queue (file_name, image_path, upload_time) VALUES (?, ?, ?)",
        (file_name, web_image_path, now_str)
    )
    conn.commit()
    conn.close()

async def process_attendance_frame(frame_bytes: bytes, websocket: WebSocket):
    img = decode_rgb565_frame(frame_bytes)
    if img is None:
        await websocket.send_json({"status": "error", "user_id_num": "FAILED"})
        return

    faces = ai_app.get(img)
    if len(faces) == 0:
        await websocket.send_json({"status": "success", "is_matched": False, "user_id_num": "UNKNOWN"})
        return

    input_vec = faces[0].embedding
    response_data = match_face_and_log(input_vec)
    
    # Gửi JSON gọn chỉ chứa ID về ESP32
    await websocket.send_json(response_data)
    # Broadcast thông báo cho Web Monitor
    await manager.broadcast(response_data)

@app.websocket("/ws/register")
async def websocket_register_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await process_register_frame(await websocket.receive_bytes(), websocket)
    except WebSocketDisconnect:
        print("ESP32 da ngat ket noi khoi dang ky.")

@app.websocket("/ws/attendance")
async def websocket_attendance_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await process_attendance_frame(await websocket.receive_bytes(), websocket)
    except WebSocketDisconnect:
        print("ESP32 da ngat ket noi khoi diem danh.")

@app.websocket("/ws/esp32")
async def websocket_esp32_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ ESP32 đã kết nối WebSocket thành công!")
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                prefix_mode = data[0]
                frame_bytes = data[1:]

                if prefix_mode == 0x01:
                    await process_register_frame(frame_bytes, websocket)

                elif prefix_mode == 0x02:
                    await process_attendance_frame(frame_bytes, websocket)
    except WebSocketDisconnect:
        print("⚠️ ESP32 đã ngắt kết nối WebSocket.")
    except RuntimeError as error:
        if 'disconnect' in str(error).lower():
            print("⚠️ ESP32 đã ngắt kết nối WebSocket.")
        else:
            raise


@app.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("🖥️ Web Monitor đã kết nối WebSocket thành công!")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("🖥️ Web Monitor đã ngắt kết nối.")


# =====================================================================
# 🌐 ROUTE WEB GIAO DIỆN HTML
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/login", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/monitor", response_class=HTMLResponse)
@app.get("/monitor.html", response_class=HTMLResponse)
async def serve_monitor(request: Request):
    return templates.TemplateResponse(request=request, name="monitor.html")

@app.get("/register", response_class=HTMLResponse)
@app.get("/register.html", response_class=HTMLResponse)
async def serve_register(request: Request):
    return templates.TemplateResponse(request=request, name="register.html")

@app.get("/storage", response_class=HTMLResponse)
@app.get("/storage.html", response_class=HTMLResponse)
async def serve_storage(request: Request):
    return templates.TemplateResponse(request=request, name="storage.html")

@app.get("/list", response_class=HTMLResponse)
@app.get("/list.html", response_class=HTMLResponse)
async def serve_list(request: Request):
    return templates.TemplateResponse(request=request, name="list.html")


# =====================================================================
# 📷 REST API CHO DỮ LIỆU VÀ QUẢN LÝ
# =====================================================================

@app.post("/api/upload-storage")
async def upload_to_storage(request: Request):
    image_bytes = await request.body()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Không có dữ liệu ảnh!"})

    _, img = decode_uploaded_image(image_bytes)
    if img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Lỗi đọc định dạng ảnh!"})

    timestamp = int(time.time())
    file_name = f"ESP32_CAP_{timestamp}.jpg"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    cv2.imwrite(file_path, img)

    web_image_path = f"/uploads/{file_name}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO storage_queue (file_name, image_path, upload_time) VALUES (?, ?, ?)",
        (file_name, web_image_path, now_str)
    )
    conn.commit()
    conn.close()

    return {
        "status": "success", 
        "message": "Đã nhận ảnh vào Storage Queue thành công!",
        "image_path": web_image_path,
        "upload_time": now_str
    }

@app.post("/api/upload-storage-file")
async def upload_storage_file(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png"}
    if file.content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Chỉ hỗ trợ ảnh JPEG hoặc PNG!"}
        )

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"status": "error", "message": "File ảnh rỗng!"})

    jpeg_bytes, img = decode_uploaded_image(image_bytes)
    if jpeg_bytes is None or img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "File ảnh không hợp lệ!"})

    file_name = f"UPLOAD_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as output_file:
        output_file.write(jpeg_bytes)

    web_image_path = f"/uploads/{file_name}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO storage_queue (file_name, image_path, upload_time) VALUES (?, ?, ?)",
        (file_name, web_image_path, now_str)
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": "Đã tải ảnh vào Storage Queue thành công!",
        "image_path": web_image_path,
        "upload_time": now_str
    }

@app.get("/api/storage-images")
async def get_storage_images():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_name, image_path, upload_time FROM storage_queue ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "file_name": row["file_name"],
            "image_path": row["image_path"],
            "upload_time": row["upload_time"]
        }
        for row in rows
    ]

@app.post("/api/upload-face")
async def upload_face_for_attendance(request: Request):
    image_bytes = await request.body()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Không có dữ liệu ảnh!"})

    jpeg_bytes, _ = decode_uploaded_image(image_bytes)
    if jpeg_bytes is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Frame RGB565 hoac anh khong hop le"})

    base64_img = base64.b64encode(jpeg_bytes).decode('ascii')
    img_data_url = f"data:image/jpeg;base64,{base64_img}"
    await manager.broadcast({"type": "stream", "image": img_data_url})

    input_vec, _ = extract_embedding_from_bytes(jpeg_bytes)
    if input_vec is None:
        result = {"status": "success", "is_matched": False, "message": "Không phát hiện khuôn mặt"}
        return JSONResponse(content=result)

    response_data = match_face_and_log(input_vec)
    await manager.broadcast(response_data)
    return JSONResponse(content=response_data)

@app.post("/api/register")
async def register_member(data: Dict[Any, Any]):
    full_name = data.get("full_name")
    id_number = data.get("id_number")
    email = data.get("email")
    position = data.get("position")
    image_path = data.get("image_path", "/uploads/default.jpg")
    register_date = data.get("register_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not full_name or not id_number:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Thiếu thông tin bắt buộc!"})

    embedding_json = json.dumps([])
    
    if image_path and image_path != "/uploads/default.jpg":
        full_file_path = os.path.join(BASE_DIR, image_path.lstrip("/"))
        if os.path.exists(full_file_path):
            img = cv2.imread(full_file_path)
            if img is not None:
                faces = ai_app.get(img)
                if len(faces) > 0:
                    embedding_json = json.dumps(faces[0].embedding.tolist())

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO storage_faces (user_id_num, user_name, email, position, register_date, image_path, embedding) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (id_number, full_name, email, position, register_date, image_path, embedding_json)
    )
    conn.commit()
    conn.close()

    return {"status": "success", "message": f"Đã đăng ký thành công thành viên {full_name}!"}

@app.get("/api/logs")
async def get_access_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id_num, full_name, email, position, access_date, accuracy, similar_to FROM access_logs ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row["id"],
            "user_id": row["user_id_num"],
            "full_name": row["full_name"],
            "email": row["email"],
            "position": row["position"],
            "access_date": row["access_date"],
            "accuracy": row["accuracy"],
            "similar_to": row["similar_to"]
        }
        for row in rows
    ]

@app.get("/api/users")
async def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id_num, user_name, email, position, register_date, image_path FROM storage_faces ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row["id"],
            "user_id": row["user_id_num"],
            "user_name": row["user_name"],
            "email": row["email"],
            "position": row["position"],
            "register_date": row["register_date"],
            "image_path": row["image_path"]
        }
        for row in rows
    ]

@app.delete("/api/storage-images/{image_id}")
async def delete_storage_image(image_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_path FROM storage_queue WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return JSONResponse(status_code=404, content={"status": "error", "message": "Không tìm thấy ảnh!"})
    
    image_path = row["image_path"]
    
    cursor.execute("DELETE FROM storage_queue WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()

    full_file_path = os.path.join(BASE_DIR, image_path.lstrip("/"))
    if os.path.exists(full_file_path):
        try:
            os.remove(full_file_path)
        except Exception as e:
            print(f"Lỗi khi xóa file: {e}")

    return {"status": "success", "message": "Đã xóa ảnh thành công!"}


if __name__ == "__main__":
    import uvicorn
    import socket

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    print("\n" + "="*50)
    print(f"🚀 SERVER ĐÃ SẴN SÀNG!")
    print(f"👉 Link Trình duyệt: http://localhost:8000")
    print(f"👉 Link IP ESP32:    ws://{local_ip}:8000/ws/esp32")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
