# app.py
# NgọcMinhChâu-AI - Full dashboard theo tài liệu (login + dashboard with full features)
# Lưu ý: cần ai_core.py (ai_pipeline) và utils.py (send_email, send_zalo_message) có sẵn
# Cài các package theo requirements.txt

import os
import io
import json
import zipfile
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv

# Load .env nếu có (local dev)
load_dotenv()

# Try imports optional
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import docx
except Exception:
    docx = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# import project modules (ai_core, utils)
try:
    from ai_core import ai_pipeline
except Exception as e:
    ai_pipeline = None
    AI_CORE_ERROR = str(e)
else:
    AI_CORE_ERROR = None

try:
    from utils import send_email, send_zalo_message
except Exception as e:
    send_email = None
    send_zalo_message = None

# ------------------------
# Config paths + env
# ------------------------
st.set_page_config(page_title="NgọcMinhChâu AI", layout="wide", page_icon="💎")
ROOT = Path.cwd()
INPUT_DIR = ROOT / "input_data"
OUTPUT_DIR = ROOT / "output_data"
HISTORY_DIR = ROOT / "report_history"
LOGS_DIR = ROOT / "logs"
for d in (INPUT_DIR, OUTPUT_DIR, HISTORY_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "history.json"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=3)

# ------------------------
# Helpers: history persistence
# ------------------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(h):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")

def add_history(rec: dict):
    h = load_history()
    h.append(rec)
    save_history(h)

# ------------------------
# Helpers: file extractors (graceful)
# ------------------------
def extract_text_from_pdf(path: Path):
    if PyPDF2 is None:
        return "[PyPDF2 chưa cài]"
    try:
        out = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    out.append(t)
        return "\n".join(out)
    except Exception as e:
        return f"[Lỗi đọc PDF] {e}"

def extract_text_from_docx(path: Path):
    if docx is None:
        return "[python-docx chưa cài]"
    try:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        return f"[Lỗi docx] {e}"

def extract_text_from_image(path: Path):
    if Image is None or pytesseract is None:
        return "[PIL/pytesseract chưa cài]"
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang="vie+eng")
    except Exception as e:
        return f"[Lỗi OCR] {e}"

def extract_text_from_audio(path: Path):
    if sr is None:
        return "[SpeechRecognition chưa cài]"
    try:
        r = sr.Recognizer()
        with sr.AudioFile(str(path)) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language="vi-VN")
    except Exception as e:
        return f"[Lỗi speech->text] {e}"

def read_excel_df(path: Path):
    if pd is None:
        return None, "[pandas chưa cài]"
    try:
        df = pd.read_excel(path)
        return df, None
    except Exception as e:
        return None, f"[Lỗi đọc Excel] {e}"

# ------------------------
# UI: small CSS & copy-to-clipboard helper
# ------------------------
st.markdown("""
<style>
.big-title { font-size:26px; font-weight:700; color:#0b3d91; }
.card { border-radius:10px; padding:14px; background: #fff; box-shadow: 0 4px 18px rgba(11, 61, 145, 0.06); }
.chat-user { background: #E6F0FF; padding:10px; border-radius:8px; }
.chat-ai { background: #F3F7FF; padding:10px; border-radius:8px; }
.small-muted { color:#6b7280; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# copy-to-clipboard: use component HTML
def copy_button_js(text, key):
    # returns HTML that copies 'text' to clipboard when clicked
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    html = f"""
    <button onclick="navigator.clipboard.writeText('{escaped}')" style="padding:6px 10px;border-radius:6px;border:none;background:#0b63c4;color:white;cursor:pointer;">
        📋 Copy
    </button>
    """
    st.components.v1.html(html, height=40, key=key)

# standardized action buttons (Copy / Delete / Download)
def action_buttons_for_text(text: str, filename: str = "output.txt", file_bytes: bytes = None, key_prefix="act"):
    cols = st.columns([1,1,1])
    with cols[0]:
        # copy using JS
        try:
            copy_button_js(text, key=key_prefix+"_copy")
            st.caption("Sao chép toàn bộ nội dung")
        except Exception:
            st.button("📋 Copy", key=key_prefix+"_copy_fallback")
    with cols[1]:
        if st.button("🗑 Xóa", key=key_prefix+"_delete"):
            return "delete"
    with cols[2]:
        if file_bytes is not None:
            st.download_button("⬇ Tải về", file_bytes, file_name=filename, key=key_prefix+"_dl")
        else:
            st.download_button("⬇ Tải về (TXT)", text, file_name=filename, key=key_prefix+"_dl_txt")
    return None

# ------------------------
# Login screen
# ------------------------
def login_ui():
    st.markdown("<div class='big-title'>NgọcMinhChâu AI — Đăng nhập</div>", unsafe_allow_html=True)
    st.write("Đăng nhập để truy cập Dashboard. (Phiên bản demo — thay bằng OAuth/SSO khi production)")
    col1, col2 = st.columns([1,1])
    with col1:
        username = st.text_input("Tài khoản")
        password = st.text_input("Mật khẩu", type="password")
        if st.button("Đăng nhập"):
            if username == ADMIN_USER and password == ADMIN_PASS:
                st.session_state['logged_in'] = True
                st.success("Đăng nhập thành công")
                st.rerun()
            else:
                st.error("Sai tài khoản hoặc mật khẩu.")
    with col2:
        st.markdown("**Hướng dẫn nhanh**")
        st.write("- Thử tài khoản mặc định (ENV ADMIN_USER/ADMIN_PASS).")
        st.write("- Không lưu API Key trong repo, đặt trên Render.")

if not st.session_state.get("logged_in", False):
    login_ui()
    st.stop()

# ------------------------
# Dashboard layout & menu
# ------------------------
st.sidebar.image("logo.png" if Path("logo.png").exists() else "https://i.ibb.co/0s3pdnc/sample-logo.png", width=130)
st.sidebar.markdown("## NgọcMinhChâu AI")
menu = st.sidebar.radio("Chọn mục:", [
    "🏠 Trang chính",
    "💬 Chat với NgocMinhChau",
    "📑 Báo cáo của tôi",
    "📜 Quy chế",
    "📝 Công văn",
    "📊 Kế hoạch",
    "🔊 Chuyển văn bản -> Giọng nói",
    "🎥 Video của tôi",
    "➕ Loại khác",
    "⚙️ Cấu hình & Gửi"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Lập trình: **Đoàn Thanh Châu** — SĐT: 0966313456")
st.sidebar.markdown("[Facebook](https://facebook.com) • [Zalo](#) • [YouTube](#) • [TikTok](#)")

# Quick state defaults
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_file_bytes" not in st.session_state:
    st.session_state.last_file_bytes = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = "output.txt"

# ------------------------
# Common: run AI safely
# ------------------------
def call_ai(prompt: str, system: str = None):
    if ai_pipeline is None:
        return f"[ai_core chưa sẵn sàng] {AI_CORE_ERROR or ''}"
    if not OPENAI_KEY:
        return "[OPENAI_API_KEY chưa được cấu hình]"
    try:
        return ai_pipeline(prompt)
    except Exception as e:
        return f"[Lỗi khi gọi AI] {e}"

# ------------------------
# Function: process uploaded file in background
# ------------------------
def background_process_file(path: Path, task_type: str = "report"):
    def job():
        txt = ""
        ext = path.suffix.lower()
        if ext == ".pdf":
            txt = extract_text_from_pdf(path)
        elif ext == ".docx":
            txt = extract_text_from_docx(path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            txt = extract_text_from_image(path)
        elif ext in (".wav", ".mp3", ".flac"):
            txt = extract_text_from_audio(path)
        elif ext in (".xls", ".xlsx", ".csv"):
            df, err = read_excel_df(path)
            if err:
                txt = err
            else:
                txt = df.to_string()
        else:
            try:
                txt = path.read_text(encoding="utf-8")
            except:
                txt = "[Không đọc được file]"

        if task_type == "report":
            prompt = f"Bạn là trợ lý hành chính. Từ nội dung sau, tạo 1 báo cáo hành chính rõ ràng, có Mục tiêu, Tóm tắt, Phân tích, Kết luận, Đề xuất:\n\n{txt[:6000]}"
            report = call_ai(prompt)
            out_path = OUTPUT_DIR / f"report_{path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            out_path.write_text(report, encoding="utf-8")
            add_history({"type":"report","source":path.name,"path":str(out_path),"timestamp":datetime.now().isoformat(),"preview":report[:800]})
        else:
            out_path = OUTPUT_DIR / f"processed_{path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            out_path.write_text(txt, encoding="utf-8")
            add_history({"type":"process","source":path.name,"path":str(out_path),"timestamp":datetime.now().isoformat(),"preview":txt[:800]})
        return str(out_path)
    fut = executor.submit(job)
    return fut

# ------------------------
# Menu handlers
# ------------------------

# --- Trang chính
if menu == "🏠 Trang chính":
    st.markdown("<div class='big-title'>NgọcMinhChâu AI — Nhiệm vụ hoàn thành trong nháy mắt!</div>", unsafe_allow_html=True)
    st.write("Hệ thống trợ giúp công chức xã: soạn công văn, báo cáo, phân tích dữ liệu, gửi thông báo.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("💬 Chat với AI")
        st.write("Trò chuyện, soạn công văn, hỗ trợ soạn thảo.")
        if st.button("Bắt đầu Chat"):
            st.session_state['nav_to'] = "💬 Chat với NgocMinhChau"
            st.rerun()
    with c2:
        st.subheader("📑 Báo cáo")
        st.write("Upload tài liệu → Tạo báo cáo tự động.")
        if st.button("Mở Báo cáo"):
            st.session_state['nav_to'] = "📑 Báo cáo của tôi"
            st.rerun()
    with c3:
        st.subheader("📊 Phân tích Excel")
        st.write("Phân tích số liệu, phát hiện bất thường.")
        if st.button("Mở Phân tích"):
            st.session_state['nav_to'] = "📊 Kế hoạch"
            st.rerun()

# --- Chat
elif menu == "💬 Chat với NgocMinhChau":
    st.header("💬 Chat trực tiếp với NgocMinhChau")
    st.markdown("Chọn chế độ: ", unsafe_allow_html=True)
    chat_mode = st.radio("Chế độ Chat:", ["Trợ lý hành chính", "Soạn công văn", "Soạn báo cáo", "Truy vấn văn bản"])
    with st.form("chat_form"):
        user_input = st.text_area("Nhập yêu cầu:", height=160)
        submitted = st.form_submit_button("Gửi")
        if submitted and user_input.strip():
            sys_prompt = {
                "Trợ lý hành chính": "Bạn là trợ lý hành chính, trả lời ngắn gọn, rõ ràng.",
                "Soạn công văn": "Bạn là chuyên gia soạn thảo công văn theo mẫu hành chính Việt Nam.",
                "Soạn báo cáo": "Bạn là chuyên gia viết báo cáo hành chính chi tiết.",
                "Truy vấn văn bản": "Bạn là trợ lý pháp lý, trả lời có dẫn nguồn (nếu có)."
            }.get(chat_mode, "")
            with st.spinner("AI đang xử lý..."):
                ans = call_ai(user_input, system=sys_prompt)
            # save to session & history
            st.session_state.chat_history.append({"role":"user","text":user_input,"ts":datetime.now().isoformat()})
            st.session_state.chat_history.append({"role":"ai","text":ans,"ts":datetime.now().isoformat()})
            add_history({"type":"chat","mode":chat_mode,"prompt":user_input,"response":ans,"timestamp":datetime.now().isoformat()})
            st.session_state.last_output = ans
            st.session_state.last_filename = f"chat_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            st.session_state.last_file_bytes = None
            st.experimental_rerun()
    st.markdown("### Phiên gần đây")
    for item in reversed(st.session_state.chat_history[-10:]):
        if item["role"] == "user":
            st.markdown(f"<div class='chat-user'>**Bạn:** {item['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>**AI:** {item['text']}</div>", unsafe_allow_html=True)
    if st.session_state.last_output:
        st.markdown("### Kết quả mới nhất")
        st.text_area("Kết quả", st.session_state.last_output, height=240)
        res = action_buttons_for_text(st.session_state.last_output, filename=st.session_state.last_filename, file_bytes=st.session_state.last_file_bytes, key_prefix="chat_last")
        if res == "delete":
            st.session_state.last_output = ""
            st.session_state.last_file_bytes = None
            st.session_state.last_filename = "output.txt"
            st.rerun()

# --- Báo cáo
elif menu == "📑 Báo cáo của tôi":
    st.header("📑 Báo cáo")
    st.write("Upload tài liệu (PDF/DOCX/Excel/Image/Audio) để AI trích xuất và sinh báo cáo.")
    uploaded = st.file_uploader("Tải lên tài liệu", accept_multiple_files=False)
    if uploaded:
        target = INPUT_DIR / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Đã lưu: {uploaded.name}")
        # trích xuất preview
        ext = target.suffix.lower()
        txt_preview = ""
        if ext == ".pdf":
            txt_preview = extract_text_from_pdf(target)
        elif ext == ".docx":
            txt_preview = extract_text_from_docx(target)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            txt_preview = extract_text_from_image(target)
        elif ext in (".wav", ".mp3", ".flac"):
            txt_preview = extract_text_from_audio(target)
        elif ext in (".xls", ".xlsx", ".csv"):
            df, err = read_excel_df(target)
            if err:
                txt_preview = err
            else:
                txt_preview = df.head(200).to_string()
        else:
            try:
                txt_preview = uploaded.getvalue().decode(errors="ignore")
            except:
                txt_preview = "[Không thể đọc file]"
        st.text_area("Nội dung trích xuất (xem nhanh)", txt_preview[:8000], height=260)
        if st.button("Tạo báo cáo bằng AI"):
            fut = background_process_file(target, task_type="report")
            st.info("Đã bắt đầu tạo báo cáo trong nền. Kiểm tra tab Lịch sử khi hoàn thành.")

    st.markdown("### Báo cáo đã tạo")
    out_files = sorted(OUTPUT_DIR.glob("report_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in out_files[:10]:
        st.markdown(f"**{f.name}** — {datetime.fromtimestamp(f.stat().st_mtime).isoformat()}")
        content = f.read_text(encoding="utf-8")
        st.text_area("Preview", content[:1500], height=180, key=f.name)
        # actions
        file_bytes = content.encode("utf-8")
        act = action_buttons_for_text(content, filename=f.name, file_bytes=file_bytes, key_prefix=f"report_{f.name}")
        if act == "delete":
            try:
                f.unlink()
                st.success("Đã xóa file.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Lỗi xóa: {e}")

# --- Quy chế
elif menu == "📜 Quy chế":
    st.header("📜 Quy chế")
    st.write("Kho văn bản quy chế — upload tài liệu quy chế để hệ thống lưu trữ và cho phép truy vấn/nghiên cứu.")
    uploaded = st.file_uploader("Upload file quy chế (PDF/DOCX)", key="quyche")
    if uploaded:
        t = INPUT_DIR / uploaded.name
        with open(t, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Đã lưu quy chế")
        add_history({"type":"quyche","path":str(t),"timestamp":datetime.now().isoformat(),"preview":uploaded.name})

# --- Công văn
elif menu == "📝 Công văn":
    st.header("📝 Soạn công văn")
    with st.form("congvan_form"):
        cv_title = st.text_input("Tiêu đề")
        cv_recipient = st.text_input("Kính gửi")
        cv_body = st.text_area("Nội dung chính", height=220)
        add_signature = st.checkbox("Chèn chữ ký / con dấu (tải ảnh)", value=False)
        signature_file = None
        if add_signature:
            signature_file = st.file_uploader("Tải ảnh chữ ký / con dấu", type=["png","jpg","jpeg"], key="sig")
        sub = st.form_submit_button("Soạn công văn")
        if sub:
            prompt = f"Soạn công văn theo mẫu hành chính Việt Nam.\nTiêu đề: {cv_title}\nKính gửi: {cv_recipient}\nNội dung: {cv_body}"
            with st.spinner("AI đang soạn công văn..."):
                doc = call_ai(prompt)
            # save
            outp = OUTPUT_DIR / f"congvan_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            outp.write_text(doc, encoding="utf-8")
            add_history({"type":"congvan","title":cv_title,"path":str(outp),"timestamp":datetime.now().isoformat(),"preview":doc[:400]})
            st.success("Đã tạo công văn")
            st.text_area("Công văn", doc, height=300)
            # if signature, append img note (for display/download user can combine externally)
            if signature_file:
                st.image(signature_file, caption="Chữ ký / con dấu", width=200)

    # actions for last created
    last_congvan = sorted(OUTPUT_DIR.glob("congvan_*.txt"), key=lambda p:p.stat().st_mtime, reverse=True)
    if last_congvan:
        f = last_congvan[0]
        cont = f.read_text(encoding="utf-8")
        st.markdown("### Công văn mới nhất")
        st.text_area("Preview", cont, height=180)
        act = action_buttons_for_text(cont, filename=f.name, file_bytes=cont.encode("utf-8"), key_prefix="cv_latest")
        if act == "delete":
            try:
                f.unlink()
                st.success("Đã xóa công văn")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi xóa: {e}")

# --- Kế hoạch
elif menu == "📊 Kế hoạch":
    st.header("📊 Kế hoạch")
    st.write("Soạn kế hoạch hoặc phân tích dữ liệu để sinh kế hoạch.")
    plan_text = st.text_area("Mô tả/ứng dụng cần lập kế hoạch", height=200)
    if st.button("Tạo kế hoạch bằng AI"):
        with st.spinner("AI đang tạo kế hoạch..."):
            plan = call_ai(f"Từ nội dung sau, lập kế hoạch chi tiết:\n\n{plan_text}")
        out = OUTPUT_DIR / f"kehoach_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        out.write_text(plan, encoding="utf-8")
        add_history({"type":"kehoach","path":str(out),"timestamp":datetime.now().isoformat(),"preview":plan[:400]})
        st.success("Đã tạo kế hoạch")
        st.text_area("Kế hoạch", plan, height=300)
        act = action_buttons_for_text(plan, filename=out.name, file_bytes=plan.encode("utf-8"), key_prefix="plan_latest")
        if act == "delete":
            out.unlink()
            st.success("Đã xóa kế hoạch")
            st.rerun()

# --- TTS
elif menu == "🔊 Chuyển văn bản -> Giọng nói":
    st.header("🔊 Text → Speech")
    tts_text = st.text_area("Nhập văn bản để chuyển sang giọng nói", height=240)
    tts_lang = st.selectbox("Ngôn ngữ", ["vi", "en"])
    if st.button("Tạo audio"):
        if gTTS is None:
            st.error("gTTS chưa cài. Cài thêm gTTS vào requirements để dùng TTS.")
        else:
            try:
                mp = io.BytesIO()
                t = gTTS(tts_text, lang=tts_lang)
                t.write_to_fp(mp)
                mp.seek(0)
                b = mp.read()
                st.audio(b, format="audio/mp3")
                st.session_state.last_file_bytes = b
                st.session_state.last_filename = f"tts_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
                add_history({"type":"tts","timestamp":datetime.now().isoformat(),"preview":tts_text[:200]})
                act = action_buttons_for_text(tts_text, filename=st.session_state.last_filename, file_bytes=b, key_prefix="tts_latest")
                if act == "delete":
                    st.session_state.last_file_bytes = None
                    st.session_state.last_filename = "output.mp3"
                    st.rerun()
            except Exception as e:
                st.error(f"Lỗi tạo audio: {e}")

# --- Video (placeholder)
elif menu == "🎥 Video của tôi":
    st.header("🎥 Quản lý Video")
    v = st.file_uploader("Upload video (hiện lưu placeholder)", type=["mp4","mov","avi"])
    if v:
        t = INPUT_DIR / v.name
        with open(t, "wb") as f:
            f.write(v.getbuffer())
        st.success("Đã lưu video. (Xử lý video nâng cao có thể thêm sau)")

# --- Loại khác
elif menu == "➕ Loại khác":
    st.header("➕ Loại khác")
    st.write("Các chức năng mở rộng: tích hợp cổng thông tin, API chính phủ, mẫu tài liệu đặc thù...")

# --- Cấu hình & gửi
elif menu == "⚙️ Cấu hình & Gửi":
    st.header("⚙️ Cấu hình & Gửi")
    st.markdown("Đặt biến môi trường trên Render hoặc .env (không push vào GitHub).")
    st.write("Các biến quan trọng: OPENAI_API_KEY, EMAIL_USER, EMAIL_PASSWORD, ZALO_ACCESS_TOKEN, ZALO_USER_ID")
    st.markdown("### Gửi thử Email / Zalo")
    col1, col2 = st.columns(2)
    with col1:
        to = st.text_input("Gửi tới (email)", value=os.getenv("EMAIL_USER",""))
        subj = st.text_input("Tiêu đề", value="Thông báo từ NgocMinhChauAI")
        body = st.text_area("Nội dung", value="Nội dung test", height=160)
        if st.button("Gửi Email thử"):
            if send_email is None:
                st.error("Hàm send_email chưa có (kiểm tra utils.py).")
            else:
                ok, msg = send_email(subj, body, to)
                if ok: st.success("Gửi email thành công")
                else: st.error(f"Lỗi gửi email: {msg}")
    with col2:
        zmsg = st.text_area("Nội dung Zalo", value="Test Zalo", height=160)
        if st.button("Gửi Zalo thử"):
            if send_zalo_message is None:
                st.error("Hàm send_zalo_message chưa có (kiểm tra utils.py).")
            else:
                ok, msg = send_zalo_message(zmsg)
                if ok: st.success("Gửi Zalo thành công")
                else: st.error(f"Lỗi gửi Zalo: {msg}")

# ------------------------
# Lịch sử & Backup (common foot)
# ------------------------
st.markdown("---")
st.markdown("### Lịch sử & Sao lưu")
history = load_history()
if history:
    st.write(f"Tổng bản ghi: {len(history)}")
    if st.button("Xóa toàn bộ lịch sử"):
        save_history([])
        st.success("Đã xóa lịch sử")
    if st.button("Tạo ZIP backup toàn bộ file output"):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        zf = OUTPUT_DIR / f"backup_reports_{ts}.zip"
        with zipfile.ZipFile(zf, "w") as z:
            for f in OUTPUT_DIR.glob("*"):
                z.write(f, arcname=f.name)
        with open(zf, "rb") as fh:
            st.download_button("Tải ZIP backup", fh, file_name=zf.name)
else:
    st.info("Hiện chưa có bản ghi nào.")

st.markdown("---")
st.markdown("© NgọcMinhChâu AI — Thiết kế theo yêu cầu. Liên hệ: Đoàn Thanh Châu — 0966313456")
