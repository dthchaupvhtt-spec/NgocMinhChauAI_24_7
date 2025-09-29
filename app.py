import streamlit as st
import os
import time
import threading
import schedule
from utils import send_email, send_zalo_message
from ai_core import ai_pipeline

st.set_page_config(page_title="NgọcMinhChâu AI", layout="wide")
st.title("NgọcMinhChâu AI - Trợ lý hành chính cá nhân")

# -----------------------------
# Nhắc nhở công việc hằng ngày
# -----------------------------
def reminder_task():
    message = "Nhắc nhở: Xem báo cáo hành chính mới."
    send_email("to_email@gmail.com", "Nhắc nhở công việc", message)
    send_zalo_message("user_id", message, "ACCESS_TOKEN")
    st.info("Đã gửi nhắc nhở Email & Zalo!")

schedule.every().day.at("08:00").do(reminder_task)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_schedule, daemon=True).start()

# -----------------------------
# Upload file
# -----------------------------
uploaded_file = st.file_uploader("Tải lên file (PDF, Excel, JPG/PNG, WAV/MP3)")

if uploaded_file is not None:
    os.makedirs("input_data", exist_ok=True)
    file_path = os.path.join("input_data", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Đang xử lý dữ liệu...")
    result, output_file = ai_pipeline(file_path)
    
    st.success("Hoàn tất! Báo cáo đã tạo.")
    
    # Hiển thị báo cáo
    st.text_area("Kết quả AI", result, height=300)
    
    # Gửi báo cáo Email & Zalo
    if st.button("Gửi báo cáo Email & Zalo"):
        send_email("to_email@gmail.com", "Báo cáo hành chính tự động", result)
        send_zalo_message("user_id", "Báo cáo hành chính mới đã được tạo!", "ACCESS_TOKEN")
        st.success("Báo cáo đã gửi qua Email & Zalo!")

# -----------------------------
# Tải file báo cáo
# -----------------------------
if os.path.exists("output_data/report.txt"):
    with open("output_data/report.txt", "r", encoding="utf-8") as f:
        report_content = f.read()
    st.download_button(
        label="Tải báo cáo AI",
        data=report_content,
        file_name="report.txt",
        mime="text/plain"
    )

