# ai_core.py
import os
from openai import OpenAI

# Lấy API key từ biến môi trường (Render -> Environment Variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ai_pipeline(prompt: str) -> str:
    """
    Hàm pipeline AI đơn giản:
    - Nhận vào prompt (câu hỏi văn bản)
    - Gửi tới OpenAI GPT-4o-mini
    - Trả về câu trả lời
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # model nhỏ, rẻ, chạy nhanh
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI thông minh và hữu ích."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi AI: {e}"

