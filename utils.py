import smtplib
from email.mime.text import MIMEText
import requests

# =========================
# Gửi Email qua Gmail SMTP
# =========================
def send_email(subject, body, to_email, from_email, password):
    """
    Hàm gửi email cơ bản.
    subject: tiêu đề email
    body: nội dung email
    to_email: người nhận
    from_email: Gmail của bạn
    password: App Password (không dùng mật khẩu Gmail thường)
    """
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        return True
    except Exception as e:
        print("Lỗi khi gửi email:", e)
        return False


# =========================
# Gửi tin nhắn qua Zalo API
# =========================
def send_zalo_message(access_token, user_id, message):
    """
    Hàm gửi tin nhắn Zalo OA.
    access_token: ZALO_ACCESS_TOKEN
    user_id: ZALO_USER_ID (ID người nhận)
    message: nội dung tin nhắn
    """
    try:
        url = "https://openapi.zalo.me/v2.0/oa/message"
        headers = {"access_token": access_token}
        data = {
            "recipient": {"user_id": user_id},
            "message": {"text": message}
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    except Exception as e:
        print("Lỗi khi gửi Zalo:", e)
        return {"error": str(e)}

