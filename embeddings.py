"""
embeddings.py
--------------
Checkpoint 2: Chunk-lar üçün embedding generasiyası

Hugging Face-in pulsuz "feature-extraction" (embedding) endpoint-indən istifadə edir.
Model: sentence-transformers/all-MiniLM-L6-v2 (kiçik, sürətli, çoxdilli dəstəyi yaxşıdır).

Endpoint: https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
(Köhnə api-inference.huggingface.co ünvanı bağlanıb, bu yeni router ünvanıdır.)
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingClient:
    """Hugging Face feature-extraction (embedding) API ilə işləmək üçün wrapper."""

    def __init__(self, model: str | None = None):
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise EnvironmentError(
                "HF_API_TOKEN tapılmadı. Zəhmət olmasa .env faylında HF_API_TOKEN təyin edin."
            )

        self.model = model or os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.api_url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self.model}/pipeline/feature-extraction"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str], max_retries: int = 3, timeout: int = 30) -> list[list[float]]:
        """
        Verilmiş mətnlər siyahısı üçün embedding vektorları qaytarır.

        Return: hər mətnə uyğun float siyahısı (vector), texts ilə eyni sırada.
        """
        if not texts:
            return []

        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    return data

                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"Status {response.status_code}: {response.text}"
                    if attempt < max_retries:
                        wait = 2 * attempt
                        print(f"[Xəbərdarlıq] Embedding sorğusu uğursuz oldu, {wait}s sonra yenidən cəhd...")
                        time.sleep(wait)
                        continue

                raise RuntimeError(f"Embedding API xətası (status {response.status_code}): {response.text}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                    continue
                raise RuntimeError(f"Embedding sorğusu şəbəkə xətası: {e}") from e

        raise RuntimeError(f"Embedding alına bilmədi: {last_error}")


def embed_chunks(chunks: list, client: EmbeddingClient | None = None, batch_size: int = 16) -> list[dict]:
    """
    Chunk siyahısı üçün embedding-lər hesablayır (batch-lərlə, çox böyük sorğuların
    qarşısını almaq üçün).

    Return: [{"chunk": Chunk, "embedding": [float, ...]}, ...]
    """
    client = client or EmbeddingClient()
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = client.embed(texts)

        for chunk, vector in zip(batch, vectors):
            results.append({"chunk": chunk, "embedding": vector})

        print(f"Embedding hesablandı: {i + len(batch)}/{len(chunks)} chunk")

    return results


if __name__ == "__main__":
    from ingest import ingest_all

    print("=== CHUNK-LARIN EMBEDDING-Ə ÇEVRİLMƏSİ ===\n")
    chunks = ingest_all(chunk_size=500, chunk_overlap=100)

    print(f"\n{len(chunks)} chunk üçün embedding hesablanır...\n")
    embedded = embed_chunks(chunks)

    print(f"\n=== NƏTİCƏ ===")
    for item in embedded:
        chunk = item["chunk"]
        vector = item["embedding"]
        print(f"{chunk} -> vektor ölçüsü: {len(vector)}, ilk 3 dəyər: {vector[:3]}")
