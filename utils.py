# utils.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# ==== GỬI EMAIL ====
def send_email(subject: str, body: str, to_email: str):
    try:
        email_user = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASSWORD")

        msg = MIMEMultipart()
        msg["From"] = email_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_user, email_password)
        server.sendmail(email_user, to_email, msg.as_string())
        server.quit()
        return "✅ Email đã được gửi thành công."
    except Exception as e:
        return f"❌ Lỗi gửi email: {e}"


# ==== GỬI TIN NHẮN ZALO ====
def send_zalo_message(message: str):
    try:
        access_token = os.getenv("ZALO_ACCESS_TOKEN")
        user_id = os.getenv("ZALO_USER_ID")
        url = "https://openapi.zalo.me/v2.0/oa/message"

        headers = {
            "access_token": access_token,
            "Content-Type": "application/json"
        }

        data = {
            "recipient": {"user_id": user_id},
            "message": {"text": message}
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return "✅ Tin nhắn Zalo đã được gửi thành công."
        else:
            return f"❌ Lỗi gửi Zalo: {response.text}"
    except Exception as e:
        return f"❌ Lỗi gửi Zalo: {e}"
