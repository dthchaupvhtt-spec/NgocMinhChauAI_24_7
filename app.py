# app.py
# NgọcMinhChâu AI — Frontend Streamlit (Phiên bản modern)
# Yêu cầu: OPENAI_API_KEY được thiết lập (env), và ai_core.py + utils.py tồn tại trong repo.

import os
from pathlib import Path
from datetime import datetime
import io
import json
import zipfile

import streamlit as st

# ----- Try imports (graceful fallback) -----
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
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
except Exception:
    np = None
    LinearRegression = None

# ----- import project modules -----
# ai_core should implement ai_pipeline(prompt: str) -> str
# utils should implement send_email(subject, body, to) and send_zalo_message(text)
try:
    from ai_core import ai_pipeline
except Exception as e:
    ai_pipeline = None
    AI_IMPORT_ERROR = str(e)
else:
    AI_IMPORT_ERROR = None

try:
    from utils import send_email, send_zalo_message
except Exception:
    send_email = None
    send_zalo_message = None

# ----- Config and folders -----
st.set_page_config(page_title="NgọcMinhChâu AI", layout="wide", page_icon="💎")
ROOT = Path.cwd()
INPUT_DIR = ROOT / "input_data"
OUTPUT_DIR = ROOT / "output_data"
HISTORY_DIR = ROOT / "report_history"
HISTORY_FILE = HISTORY_DIR / "history.json"

for d in (INPUT_DIR, OUTPUT_DIR, HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_CONFIGURED = bool(os.getenv("OPENAI_API_KEY", ""))

# ----- CSS styling for nicer UI -----
st.markdown(
    """
    <style>
    /* page */
    .big-title { font-size:28px; font-weight:700; color:#0f172a; }
    .muted { color:#6b7280; }
    .card { border-radius:12px; padding:16px; background:linear-gradient(180deg,#ffffff,#fbfdff); box-shadow:0 6px 18px rgba(15,23,42,0.06); }
    .side-title { font-size:18px; font-weight:600; color:#0f172a; }
    .small-muted { color:#94a3b8; font-size:12px; }
    .chat-user { background:#dbeafe; padding:10px; border-radius:10px; }
    .chat-ai { background:#eef2ff; padding:10px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Utilities: history -----
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(items):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def add_history(record: dict):
    h = load_history()
    h.append(record)
    save_history(h)

# ----- File extractors -----
def extract_text_from_pdf(path: Path):
    if PyPDF2 is None:
        return "[PyPDF2 chưa cài]"
    try:
        texts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
        return "\n".join(texts)
    except Exception as e:
        return f"[Lỗi đọc PDF] {e}"

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

def extract_text_from_docx(path: Path):
    try:
        import docx
    except Exception:
        return "[python-docx chưa cài]"
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[Lỗi docx] {e}"

def read_excel_as_df(path: Path):
    if pd is None:
        return None, "[pandas chưa cài]"
    try:
        df = pd.read_excel(path)
        return df, None
    except Exception as e:
        return None, f"[Lỗi đọc Excel] {e}"

# ----- Simple forecast & anomaly detection -----
def forecast_and_detect(df):
    if LinearRegression is None or np is None:
        return {}, {}
    numeric = df.select_dtypes(include="number").columns
    forecasts = {}
    alerts = {}
    for col in numeric:
        series = df[col].dropna().values
        if len(series) < 4:
            continue
        X = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(X, series)
        pred = float(model.predict([[len(series)]])[0])
        forecasts[col] = pred
        mean = float(np.mean(series)); std = float(np.std(series))
        anomalies = []
        for i, val in enumerate(series):
            if abs(val - mean) > 2 * std:
                anomalies.append({"index": int(i), "value": float(val)})
        if anomalies:
            alerts[col] = anomalies
    return forecasts, alerts

# ----- Tiny semantic store (in-memory) - optional -----
VECTOR_STORE = []

def add_to_vector_store(id_, text, embedding=None):
    VECTOR_STORE.append({"id": id_, "text": text, "embedding": embedding})

# ----- Sidebar -----
with st.sidebar:
    st.image("logo.png" if Path("logo.png").exists() else "https://i.ibb.co/0s3pdnc/sample-logo.png", width=140)
    st.markdown("<div class='side-title'>NgọcMinhChâu AI — Trợ lý hành chính</div>", unsafe_allow_html=True)
    st.write("Hỗ trợ: soạn công văn, báo cáo, tổng hợp số liệu, gửi thông báo.")
    st.markdown("---")
    st.markdown("### Cấu hình nhanh")
    st.write("OpenAI:", "✅" if OPENAI_CONFIGURED else "❌")
    if not OPENAI_CONFIGURED:
        st.warning("OpenAI API chưa cấu hình. Vui lòng đặt OPENAI_API_KEY trong Environment.")
    st.markdown("---")
    if st.button("🔄 Reset phiên làm việc"):
        # reset session variables related to chat
        keys = [k for k in st.session_state.keys() if k.startswith("chat_") or k in ("uploaded_file",)]
        for k in keys:
            del st.session_state[k]
        st.success("Đã reset phiên. Refresh trang.")
        st.rerun()
    st.markdown("---")
    st.markdown("<div class='small-muted'>Phiên bản demo — Liên hệ để triển khai nâng cao</div>", unsafe_allow_html=True)

# ----- Main layout: tabs -----
tabs = st.tabs(["🏠 Trang chủ", "💬 Chat AI", "📂 Upload & Xử lý", "📈 Phân tích Excel", "📜 Lịch sử", "⚙️ Cấu hình & Gửi"])

# ----- Tab: Home -----
with tabs[0]:
    st.markdown("<div class='big-title'>NgọcMinhChâu AI</div>", unsafe_allow_html=True)
    st.write("Trợ lý AI chuyên cho công chức xã — tự động hóa văn bản, tổng hợp báo cáo, phân tích dữ liệu và phân phối văn bản.")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.markdown("#### Thử lệnh nhanh")
        sample = st.selectbox("Chọn mẫu", [
            "Soạn công văn thông báo họp ngày ...",
            "Tóm tắt biên bản cuộc họp",
            "Soạn kế hoạch triển khai chuyển đổi số"
        ])
        q = st.text_input("Hoặc nhập yêu cầu:", value=sample)
        if st.button("Gửi lệnh (Nhanh)"):
            if ai_pipeline is None:
                st.error(f"AI chưa sẵn sàng: {AI_IMPORT_ERROR}")
            else:
                with st.spinner("AI đang xử lý..."):
                    ans = ai_pipeline(q)
                st.success("Hoàn tất")
                st.code(ans)
    with c2:
        st.markdown("#### Tài nguyên nhanh")
        st.write("- Upload file: PDF / Word / Excel / Image / Audio")
        st.write("- Xem Lịch sử, Backup, Tải báo cáo")
    with c3:
        st.markdown("#### Trạng thái")
        st.write(f"- OpenAI key: {'OK' if OPENAI_CONFIGURED else 'Chưa'}")
        st.write(f"- Files trong input_data: {len(list(INPUT_DIR.iterdir()))}")

# ----- Tab: Chat AI -----
with tabs[1]:
    st.subheader("Chat trực tiếp với AI")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_col, info_col = st.columns([3,1])
    with chat_col:
        prompt = st.text_area("Nhập câu hỏi / yêu cầu:", key="chat_input", height=140)
        if st.button("Gửi yêu cầu"):
            if not ai_pipeline:
                st.error("Hàm ai_pipeline chưa được cấu hình hoặc import thất bại.")
            elif not OPENAI_CONFIGURED:
                st.error("OPENAI_API_KEY chưa cấu hình.")
            else:
                with st.spinner("Gọi AI..."):
                    response = ai_pipeline(prompt)
                st.session_state.chat_history.append({"role":"user","text":prompt,"response":response,"ts":datetime.now().isoformat()})
                add_history({"type":"chat","prompt":prompt,"response":response,"timestamp":datetime.now().isoformat()})
                st.experimental_set_query_params()  # harmless - just to update URL if needed
                st.rerun()
    with info_col:
        st.markdown("**Phiên gần đây**")
        for item in reversed(st.session_state.get("chat_history", [])[-6:]):
            st.markdown(f"- **Bạn:** {item['text']}")
            st.markdown(f"  - **AI:** {item['response'][:200]}...")

    # display chat conversation
    st.markdown("----")
    for i, item in enumerate(st.session_state.get("chat_history", [])):
        if item["role"] == "user":
            st.markdown(f"<div class='chat-user'>{item['text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-ai small-muted'>{item['response']}</div>", unsafe_allow_html=True)

# ----- Tab: Upload & Processing -----
with tabs[2]:
    st.subheader("Upload file để AI xử lý")
    uploaded = st.file_uploader("Chọn file (PDF / Image / Excel / DOCX / Audio)", accept_multiple_files=False)
    if uploaded:
        target = INPUT_DIR / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Đã lưu: {uploaded.name}")
        st.markdown("### Trích xuất nội dung")
        txt = ""
        if uploaded.name.lower().endswith(".pdf"):
            txt = extract_text_from_pdf(target)
        elif uploaded.name.lower().endswith((".png",".jpg",".jpeg",".tiff")):
            txt = extract_text_from_image(target)
        elif uploaded.name.lower().endswith((".wav",".mp3",".flac")):
            txt = extract_text_from_audio(target)
        elif uploaded.name.lower().endswith(".docx"):
            txt = extract_text_from_docx(target)
        elif uploaded.name.lower().endswith((".xls",".xlsx",".csv")):
            df, err = read_excel_as_df(target)
            if err:
                txt = err
            else:
                txt = df.to_string()
        else:
            try:
                txt = uploaded.getvalue().decode(errors="ignore")
            except Exception:
                txt = "[Không thể giải mã file]"

        st.text_area("Nội dung trích xuất (xem nhanh)", txt[:4000], height=260)

        # Lưu tạm và tạo báo cáo
        if st.button("Tạo báo cáo từ file"):
            if not ai_pipeline:
                st.error("ai_pipeline chưa sẵn sàng.")
            else:
                with st.spinner("AI tạo báo cáo..."):
                    prompt = f"Bạn là trợ lý hành chính. Từ nội dung sau, soạn thành báo cáo hành chính rõ ràng:\n\n{txt[:6000]}"
                    if st.session_state.get("style_note"):
                        prompt = st.session_state["style_note"] + "\n\n" + prompt
                    report = ai_pipeline(prompt)
                out_name = OUTPUT_DIR / f"report_{uploaded.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
                out_name.write_text(report, encoding="utf-8")
                add_history({"type":"report","source":uploaded.name,"path":str(out_name),"timestamp":datetime.now().isoformat(),"preview":report[:800]})
                st.success("Đã tạo báo cáo")
                st.download_button("Tải báo cáo (TXT)", report, file_name=out_name.name)
                st.text_area("Báo cáo (chi tiết)", report, height=360)

# ----- Tab: Excel Analysis -----
with tabs[3]:
    st.subheader("Phân tích Excel: biểu đồ, dự báo & phát hiện bất thường")
    excel_files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".xlsx",".xls",".csv")]
    if len(excel_files) == 0:
        st.info("Chưa có file Excel trong input_data. Upload trong tab Upload & Xử lý.")
    else:
        sel = st.selectbox("Chọn file Excel", [p.name for p in excel_files])
        df = pd.read_excel(INPUT_DIR / sel) if pd else None
        if df is None:
            st.error("Pandas chưa cài; không thể phân tích.")
        else:
            st.dataframe(df.head(200))
            if st.button("Phân tích & Dự báo"):
                with st.spinner("Đang phân tích..."):
                    forecasts, alerts = forecast_and_detect(df)
                st.markdown("#### Dự báo")
                st.json(forecasts)
                if alerts:
                    st.warning("Phát hiện bất thường")
                    st.json(alerts)
                # Vẽ biểu đồ (numeric first)
                nums = df.select_dtypes(include="number").columns.tolist()
                if nums and plt:
                    fig, ax = plt.subplots()
                    ax.plot(df[nums[0]].fillna(method="ffill").values)
                    ax.set_title(nums[0])
                    st.pyplot(fig)

# ----- Tab: History -----
with tabs[4]:
    st.subheader("Lịch sử xử lý")
    history = load_history()
    if not history:
        st.info("Chưa có lịch sử.")
    else:
        for i, rec in enumerate(reversed(history[-100:]), 1):
            st.markdown(f"**{i}.** `{rec.get('type','?')}` — {rec.get('timestamp')}")
            if rec.get("preview"):
                st.write(rec.get("preview"))
            if rec.get("path") and Path(rec["path"]).exists():
                with open(rec["path"], "rb") as fh:
                    st.download_button("Tải file", fh, file_name=Path(rec["path"]).name)

    if st.button("Tạo backup ZIP tất cả báo cáo"):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        zname = OUTPUT_DIR / f"backup_reports_{ts}.zip"
        with zipfile.ZipFile(zname, "w") as z:
            for f in OUTPUT_DIR.glob("report_*.txt"):
                z.write(f, arcname=f.name)
        with open(zname, "rb") as fh:
            st.download_button("Tải ZIP backup", fh, file_name=zname.name)

# ----- Tab: Config & Send -----
with tabs[5]:
    st.subheader("Cấu hình & Gửi thông báo")
    st.write("Các biến môi trường (Render / .env): OPENAI_API_KEY, EMAIL_USER, EMAIL_PASSWORD, ZALO_ACCESS_TOKEN, ZALO_USER_ID")
    st.markdown("### Gửi Email thử")
    col1, col2 = st.columns(2)
    with col1:
        to = st.text_input("Địa chỉ email nhận", value=os.getenv("EMAIL_USER",""))
        subject = st.text_input("Tiêu đề", value="Thông báo từ NgọcMinhChauAI")
        body = st.text_area("Nội dung", value="Nội dung test", height=140)
        if st.button("Gửi Email thử"):
            if send_email:
                ok, msg = send_email(subject, body, to)
                if ok:
                    st.success("Gửi email thành công")
                else:
                    st.error(f"Gửi email lỗi: {msg}")
            else:
                st.error("Hàm gửi email chưa được cấu hình (utils.send_email).")
    with col2:
        zalo_msg = st.text_area("Nội dung Zalo", value="Test Zalo", height=140)
        if st.button("Gửi Zalo thử"):
            if send_zalo_message:
                ok = send_zalo_message(zalo_msg)
                st.write(ok)
            else:
                st.error("Hàm gửi Zalo chưa được cấu hình (utils.send_zalo_message).")

st.markdown("---")
st.markdown("© NgọcMinhChâu AI — Demo. Liên hệ để tùy biến & triển khai production.")
