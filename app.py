# app.py
# NgọcMinhChâu-AI - Streamlit front-end (đầy đủ chức năng)
# Yêu cầu: đã cài requirements, đã set OPENAI_API_KEY trong env

import os
import io
import zipfile
import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# tải env
load_dotenv()

# libs optional, try import and provide graceful fallback messages
try:
    import openai
except Exception:
    openai = None

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
    from sklearn.linear_model import LinearRegression
    import numpy as np
except Exception:
    LinearRegression = None
    np = None

# Config
st.set_page_config(page_title="NgọcMinhChâu AI — Trợ lý hành chính", layout="wide", initial_sidebar_state="expanded")
ROOT = Path.cwd()
INPUT_DIR = ROOT / "input_data"
OUTPUT_DIR = ROOT / "output_data"
HISTORY_FILE = ROOT / "report_history" / "history.json"
for d in (INPUT_DIR, OUTPUT_DIR, ROOT / "report_history", ROOT / "logs"):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ----------------------
# Helper: UI styling
# ----------------------
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#0f172a,#0b1220);
        color: white;
    }
    .big-title {
        font-size:28px;
        font-weight:700;
        color:#073b4c;
    }
    .muted {
        color:#6b7280;
    }
    .card {
        border-radius:10px;
        padding:16px;
        background: linear-gradient(90deg, #ffffff, #f8fafc);
        box-shadow: 0 4px 12px rgba(16,24,40,0.06);
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Helper: utils
# ----------------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def add_history(item: dict):
    h = load_history()
    h.append(item)
    save_history(h)

def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".xlsx", ".xls", ".csv", ".txt", ".docx", ".wav", ".mp3", ".flac"))

# ----------------------
# AI core simplified
# ----------------------
def openai_chat(prompt, system="Bạn là trợ lý hành chính chuyên nghiệp, viết cho công chức xã. Trả lời ngắn gọn, rõ ràng."):
    if not openai:
        return "OpenAI library chưa cài. Thêm openai vào requirements."
    if not OPENAI_API_KEY:
        return "ENV OPENAI_API_KEY chưa được cấu hình."
    try:
        # use ChatCompletion
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        return f"[Lỗi OpenAI] {e}"

def summarize_text(text, style="Tóm tắt ngắn gọn (báo cáo hành chính)"):
    prompt = f"Please summarize and convert the following text into a {style} in Vietnamese. Provide clear sections: Mục tiêu, Tóm tắt, Phân tích, Đề xuất.\n\nText:\n{text[:4000]}"
    return openai_chat(prompt)

# ----------------------
# File extractors
# ----------------------
def extract_text_from_pdf(path):
    if not PyPDF2:
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
        return f"[Lỗi PDF] {e}"

def extract_text_from_image(path):
    if not pytesseract or not Image:
        return "[PIL/pytesseract chưa cài]"
    try:
        img = Image.open(path)
        txt = pytesseract.image_to_string(img, lang="vie+eng")
        return txt
    except Exception as e:
        return f"[Lỗi OCR] {e}"

def extract_text_from_audio(path):
    if not sr:
        return "[SpeechRecognition chưa cài]"
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as src:
            audio = r.record(src)
        txt = r.recognize_google(audio, language="vi-VN")
        return txt
    except Exception as e:
        return f"[Lỗi Speech->Text] {e}"

def extract_text_from_excel(path):
    if not pd:
        return "[pandas chưa cài]"
    try:
        df = pd.read_excel(path)
        return df, df.to_string()
    except Exception as e:
        return None, f"[Lỗi đọc Excel] {e}"

# ----------------------
# Forecast & anomaly (simple)
# ----------------------
def forecast_and_detect(df):
    if LinearRegression is None or np is None:
        return {}, {}
    numeric_cols = df.select_dtypes(include="number").columns
    forecasts = {}
    alerts = {}
    for c in numeric_cols:
        series = df[c].dropna().values
        if len(series) < 4:
            continue
        X = np.arange(len(series)).reshape(-1,1)
        model = LinearRegression().fit(X, series)
        pred = model.predict([[len(series)]])[0]
        forecasts[c] = float(pred)
        mean = float(np.mean(series)); std = float(np.std(series))
        anomalies = []
        for i, val in enumerate(series):
            if abs(val-mean) > 2*std:
                anomalies.append({"index": int(i), "value": float(val)})
        if anomalies:
            alerts[c] = anomalies
    return forecasts, alerts

# ----------------------
# Email & Zalo utilities
# ----------------------
import smtplib
from email.mime.text import MIMEText
import requests

def send_email_simple(subject, body, to_email=None):
    user = os.getenv("EMAIL_USER")
    pwd = os.getenv("EMAIL_PASSWORD")
    if not user or not pwd:
        return False, "EMAIL_USER/EMAIL_PASSWORD chưa cấu hình"
    if not to_email:
        to_email = user
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_email
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20)
        server.login(user, pwd)
        server.sendmail(user, [to_email], msg.as_string())
        server.quit()
        return True, "Gửi email thành công"
    except Exception as e:
        return False, str(e)

def send_zalo_simple(text):
    token = os.getenv("ZALO_ACCESS_TOKEN")
    user_id = os.getenv("ZALO_USER_ID")
    if not token or not user_id:
        return False, "Zalo token/uid chưa cấu hình"
    payload = {
        "recipient": {"user_id": user_id},
        "message": {"text": text},
        "access_token": token
    }
    try:
        r = requests.post("https://openapi.zalo.me/v2.0/oa/message", json=payload, timeout=10)
        return (r.status_code == 200), r.text
    except Exception as e:
        return False, str(e)

# ----------------------
# Simple semantic store (in-memory) using OpenAI embeddings if available
# ----------------------
VECTOR_STORE = []  # list of dict {id, text, embedding}

def embed_text(text):
    if not openai:
        return None
    try:
        emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return emb["data"][0]["embedding"]
    except Exception:
        return None

def add_to_store(id_, text):
    emb = embed_text(text) if openai else None
    VECTOR_STORE.append({"id": id_, "text": text, "embedding": emb})

def semantic_search(query, top_k=3):
    if not openai or not VECTOR_STORE:
        return []
    q_emb = embed_text(query)
    if q_emb is None:
        return []
    # cosine similarity
    def cos(a,b):
        a = np.array(a); b = np.array(b)
        return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    scored = []
    for item in VECTOR_STORE:
        if item["embedding"] is None:
            continue
        s = cos(q_emb, item["embedding"])
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

# ----------------------
# Streamlit App Layout
# ----------------------
st.markdown("<div class='big-title'>NgọcMinhChâu AI — Trợ lý hành chính thông minh</div>", unsafe_allow_html=True)
st.write("Giao diện tối giản & chuyên nghiệp — Upload file, Chat, Tóm tắt, Phân tích số liệu, Gửi thông báo.")

# Sidebar controls
with st.sidebar:
    st.header("Cài đặt nhanh")
    st.write("Trạng thái OpenAI:", "OK" if OPENAI_API_KEY and openai else "Chưa cấu hình")
    if st.button("Xem lịch sử xử lý"):
        st.experimental_set_query_params(tab="history")
    st.markdown("---")
    # Personalization: save writing style
    if "style_note" not in st.session_state:
        st.session_state.style_note = ""
    st.text_area("Ghi chú phong cách (ví dụ: trang trọng, ngắn gọn, dễ hiểu):", key="style_note", height=80)
    if st.button("Lưu phong cách"):
        st.success("Đã lưu phong cách cá nhân.")
    st.markdown("---")
    st.write("Phiên bản: 1.0 | Được thiết kế cho công chức xã")

# Main Tabs
tabs = st.tabs(["🏠 Trang chủ", "💬 Chat AI", "📂 Upload & Xử lý", "📈 Phân tích Excel", "📜 Lịch sử", "⚙️ Cấu hình & Gửi"])

# --- Tab Home
with tabs[0]:
    st.subheader("Chào mừng đến với NgọcMinhChâu AI")
    st.info("Mục tiêu: hỗ trợ soạn thảo văn bản, tổng hợp báo cáo, phân tích số liệu và tự động hóa quy trình hành chính.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Nhanh: thử một lệnh")
        q = st.text_input("Hỏi AI một câu (ví dụ: Soạn công văn thông báo họp 10/10):")
        if st.button("Gửi lệnh"):
            prompt = f"{st.session_state.get('style_note','')} \n\n{q}"
            with st.spinner("AI đang suy nghĩ..."):
                ans = openai_chat(prompt)
            st.markdown("**Kết quả**")
            st.write(ans)
    with col2:
        st.markdown("### Tài nguyên nhanh")
        st.write("- Upload file (tab Upload).")
        st.write("- Xem lịch sử, tải bản sao lưu (tab Lịch sử).")
        st.write("- Cấu hình email/Zalo để gửi tự động (tab Cấu hình).")

# --- Tab Chat AI
with tabs[1]:
    st.subheader("Chat trực tiếp với AI")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    col_a, col_b = st.columns([3,1])
    with col_a:
        prompt = st.text_area("Nhập câu hỏi / yêu cầu:", height=140)
        if st.button("Gửi yêu cầu"):
            full_prompt = (st.session_state.get("style_note","") + "\n\n" + prompt).strip()
            with st.spinner("Gọi OpenAI..."):
                resp = openai_chat(full_prompt)
            st.session_state.chat_history.append({"role":"user","text":prompt,"response":resp,"ts":datetime.now().isoformat()})
            add_history({"type":"chat","prompt":prompt,"response":resp,"timestamp":datetime.now().isoformat()})
            st.experimental_rerun()
    with col_b:
        st.markdown("**Phiên gần đây**")
        for c in reversed(st.session_state.chat_history[-5:]):
            st.write(f"> **Bạn:** {c['text']}")
            st.write(f"**AI:** {c['response']}")

# --- Tab Upload & Processing
with tabs[2]:
    st.subheader("Upload file để AI xử lý")
    uploaded = st.file_uploader("Chọn file (PDF, Image, Excel, Audio, DOCX)", accept_multiple_files=False)
    if uploaded:
        save_path = INPUT_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Đã lưu: {save_path.name}")
        # extract text based on type
        txt = ""
        out_preview = ""
        if uploaded.name.lower().endswith(".pdf"):
            txt = extract_text_from_pdf(save_path)
            out_preview = txt[:2000]
        elif uploaded.name.lower().endswith((".png",".jpg",".jpeg",".tiff")):
            txt = extract_text_from_image(save_path)
            out_preview = txt[:2000]
        elif uploaded.name.lower().endswith((".wav",".mp3",".flac")):
            txt = extract_text_from_audio(save_path)
            out_preview = txt[:2000]
        elif uploaded.name.lower().endswith((".xls",".xlsx",".csv")):
            df, txt = extract_text_from_excel(save_path)
            out_preview = txt[:2000] if isinstance(txt,str) else str(df.head())
        elif uploaded.name.lower().endswith(".docx"):
            try:
                import docx
                doc = docx.Document(save_path)
                txt = "\n".join(p.text for p in doc.paragraphs)
                out_preview = txt[:2000]
            except Exception as e:
                txt = f"[Lỗi đọc docx] {e}"
        else:
            txt = uploaded.getvalue().decode(errors="ignore")
            out_preview = txt[:2000]

        st.markdown("### Nội dung trích xuất (xem nhanh)")
        st.text_area("Preview", out_preview, height=250)

        # semantic store add
        add_to_store(str(datetime.now().timestamp()), txt[:5000])

        # generate report
        st.markdown("### Tạo báo cáo / tóm tắt")
        report_style = st.selectbox("Chọn kiểu báo cáo", ["Báo cáo hành chính ngắn gọn", "Báo cáo chi tiết", "Biên bản họp", "Công văn", "Kế hoạch"])
        if st.button("Tạo báo cáo bằng AI"):
            with st.spinner("AI đang tạo báo cáo..."):
                prompt = f"Bạn là trợ lý hành chính. Từ nội dung sau, soạn thành một {report_style} bằng tiếng Việt, rõ ràng, có mục lục, các đề xuất hành động:\n\n{txt[:6000]}"
                if st.session_state.style_note:
                    prompt = st.session_state.style_note + "\n\n" + prompt
                report = openai_chat(prompt)
                # save file
                tfn = OUTPUT_DIR / f"report_{uploaded.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
                tfn.write_text(report, encoding="utf-8")
                add_history({"type":"report","source":uploaded.name,"path":str(tfn),"timestamp":datetime.now().isoformat(),"preview":report[:1000]})
                st.success("Đã tạo báo cáo")
                st.download_button("Tải báo cáo (TXT)", report, file_name=tfn.name)
                st.text_area("Báo cáo", report, height=360)

# --- Tab Excel Analysis
with tabs[3]:
    st.subheader("Phân tích Excel: dự báo & phát hiện bất thường")
    excel_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in (".xlsx",".xls",".csv")]
    if excel_files:
        sel = st.selectbox("Chọn file Excel:", [f.name for f in excel_files])
        df = pd.read_excel(INPUT_DIR / sel)
        st.dataframe(df)
        if st.button("Phân tích & Dự báo"):
            with st.spinner("Đang phân tích..."):
                forecasts, alerts = forecast_and_detect(df)
                st.markdown("#### Dự báo")
                st.json(forecasts)
                if alerts:
                    st.warning("Phát hiện bất thường")
                    st.json(alerts)
                # chart first numeric column
                numeric = df.select_dtypes(include="number").columns.tolist()
                if numeric and plt:
                    col = numeric[0]
                    fig, ax = plt.subplots()
                    ax.plot(df[col].fillna(method="ffill").values)
                    ax.set_title(f"Biểu đồ {col}")
                    st.pyplot(fig)
    else:
        st.info("Chưa có file Excel trong input_data. Upload ở tab Upload & Xử lý.")

# --- Tab History
with tabs[4]:
    st.subheader("Lịch sử & Backup")
    history = load_history()
    if not history:
        st.info("Chưa có lịch sử nào.")
    else:
        st.write(f"Tổng bản ghi: {len(history)}")
        for i, rec in enumerate(reversed(history[-50:]), 1):
            st.markdown(f"**{i}.** `{rec.get('type','?')}` — {rec.get('timestamp')}")
            if rec.get("preview"):
                st.write(rec.get("preview"))
            if rec.get("path") and Path(rec["path"]).exists():
                with open(rec["path"], "rb") as fh:
                    st.download_button("Tải file", fh, file_name=Path(rec["path"]).name)

    # backup all outputs
    if st.button("Tạo backup ZIP tất cả báo cáo"):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        zname = OUTPUT_DIR / f"backup_reports_{ts}.zip"
        with zipfile.ZipFile(zname, "w") as z:
            for f in OUTPUT_DIR.glob("report_*.txt"):
                z.write(f, arcname=f.name)
        with open(zname, "rb") as fh:
            st.download_button("Tải ZIP backup", fh, file_name=zname.name)

# --- Tab Config & Send
with tabs[5]:
    st.subheader("Cấu hình & Gửi thông báo")
    st.markdown("### Cấu hình email/Zalo (những biến này nên đặt trong Environment trên Render hoặc .env cục bộ)")
    st.write("EMAIL_USER, EMAIL_PASSWORD (App Password Gmail), ZALO_ACCESS_TOKEN, ZALO_USER_ID")
    st.markdown("### Gửi thử email / Zalo")
    col1, col2 = st.columns(2)
    with col1:
        to = st.text_input("Gửi đến (email)", value=os.getenv("EMAIL_USER",""))
        subject = st.text_input("Tiêu đề", value="Thông báo từ NgọcMinhChâu AI")
        body = st.text_area("Nội dung", value="Nội dung test", height=120)
        if st.button("Gửi Email thử"):
            ok, msg = send_email_simple(subject, body, to)
            if ok:
                st.success("Gửi email thành công")
            else:
                st.error(f"Gửi email thất bại: {msg}")
    with col2:
        zalo_msg = st.text_area("Nội dung Zalo", value="Test Zalo", height=120)
        if st.button("Gửi Zalo thử"):
            ok, msg = send_zalo_simple(zalo_msg)
            if ok:
                st.success("Gửi Zalo thành công")
            else:
                st.error(f"Gửi Zalo thất bại: {msg}")

    st.markdown("---")
    st.markdown("### Tìm kiếm ngữ nghĩa (semantic search)")
    q = st.text_input("Nhập truy vấn tìm kiếm ngữ nghĩa")
    if st.button("Tìm kiếm"):
        results = semantic_search(q)
        if not results:
            st.info("Chưa có dữ liệu embedding hoặc OpenAI không cấu hình.")
        else:
            for r in results:
                st.markdown(f"- **{r['id']}**: {r['text'][:300]}...")

# Footer
st.markdown("---")
st.write("© NgọcMinhChâu AI — Phiên bản demo. Liên hệ để triển khai nâng cao: cấu hình vector DB, xác thực user, giao diện chuyên nghiệp hơn.")

