import streamlit as st
from streamlit_chat import message
from ai_core import ai_pipeline
from utils import send_email, send_zalo_message

# ========== Cấu hình trang ==========
st.set_page_config(
    page_title="NgọcMinhChâu AI",
    page_icon="💎",
    layout="wide"
)

# ========== Thanh sidebar ==========
with st.sidebar:
    st.image("https://i.ibb.co/d64KgCT/assistant.png", width=120)
    st.title("💎 NgọcMinhChâu AI")
    menu = st.radio("📌 Chọn tính năng:", [
        "🤖 Trò chuyện AI",
        "📊 Sinh báo cáo",
        "📂 Kho dữ liệu",
        "📧 Gửi Email/Zalo",
        "⚙️ Cài đặt"
    ])

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🌐 Trợ lý AI hiện đại - WinX</h1>", unsafe_allow_html=True)
st.markdown("---")

# ========== 1. Trò chuyện AI ==========
if menu == "🤖 Trò chuyện AI":
    st.subheader("💬 Hỏi đáp & Soạn thảo văn bản")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("✍️ Nhập câu hỏi hoặc yêu cầu của bạn:")
    if st.button("Gửi"):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            ai_response = ai_pipeline(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    for i, chat in enumerate(st.session_state.chat_history):
        message(chat["content"], is_user=(chat["role"] == "user"), key=str(i))

# ========== 2. Sinh báo cáo ==========
elif menu == "📊 Sinh báo cáo":
    st.subheader("📑 Tạo báo cáo, công văn tự động")
    title = st.text_input("📝 Tiêu đề báo cáo")
    content = st.text_area("📄 Nội dung chính")
    if st.button("📌 Sinh báo cáo"):
        if title and content:
            full_report = ai_pipeline(f"Tạo báo cáo với tiêu đề '{title}' và nội dung: {content}")
            st.success("✅ Đã tạo báo cáo!")
            st.download_button("⬇️ Tải báo cáo (TXT)", full_report, file_name="bao_cao.txt")

# ========== 3. Kho dữ liệu ==========
elif menu == "📂 Kho dữ liệu":
    st.subheader("📂 Quản lý tài liệu & dữ liệu")
    uploaded_file = st.file_uploader("📤 Tải tài liệu lên", type=["txt", "pdf", "docx"])
    if uploaded_file:
        st.success(f"✅ Đã tải lên: {uploaded_file.name}")
        st.write("📖 Nội dung xem trước:")
        st.text(uploaded_file.getvalue()[:500])  # preview

# ========== 4. Gửi Email/Zalo ==========
elif menu == "📧 Gửi Email/Zalo":
    st.subheader("📧 Gửi thông báo nhanh")
    option = st.selectbox("Chọn kênh gửi:", ["Email", "Zalo"])
    message_text = st.text_area("✍️ Nội dung tin nhắn")
    if st.button("📨 Gửi ngay"):
        if option == "Email":
            send_email("NgocMinhChauAI - Thông báo", message_text)
            st.success("✅ Đã gửi email!")
        elif option == "Zalo":
            send_zalo_message(message_text)
            st.success("✅ Đã gửi tin nhắn Zalo!")

# ========== 5. Cài đặt ==========
elif menu == "⚙️ Cài đặt":
    st.subheader("⚙️ Cấu hình hệ thống")
    st.info("🔑 Bạn có thể cấu hình API Key, Email, Zalo trong file `.env`")
    st.code("""
OPENAI_API_KEY=xxxx
EMAIL_USER=xxxx
EMAIL_PASSWORD=xxxx
ZALO_ACCESS_TOKEN=xxxx
ZALO_USER_ID=xxxx
    """, language="bash")

