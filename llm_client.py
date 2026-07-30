"""
llm_client.py
--------------
RAG pipeline üçün sadə LLM chat client-i (Hugging Face router API, OpenAI-uyğun format).
Bax: Həftə 1-dəki hf_client.py-ın sadələşdirilmiş versiyası.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"


def chat(system_prompt: str, user_prompt: str, model: str | None = None, temperature: float = 0.3) -> str:
    """Hugging Face router API-yə sorğu göndərir, mətn cavabı qaytarır."""
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise EnvironmentError("HF_API_TOKEN tapılmadı. .env faylını yoxlayın.")

    model = model or os.getenv("HF_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 400,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"LLM API xətası (status {response.status_code}): {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
