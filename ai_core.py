# ai_core.py
import os
from openai import OpenAI

# Lấy API key từ biến môi trường
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ai_pipeline(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    """
    Gửi prompt đến OpenAI GPT và trả về kết quả.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI hành chính NgọcMinhChâu."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi OpenAI API: {e}"
