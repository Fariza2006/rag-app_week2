"""
vector_store.py
-----------------
Checkpoint 3: Vektor saxlama + oxşarlıq axtarışı

Chroma (lokal, server tələb etməyən vektor verilənlər bazası) istifadə edir.
Chunk-ları və onların embedding-lərini saxlayır, sonra istifadəçi sualına ən
oxşar chunk-ları tapır (cosine similarity ilə, Chroma-nın daxili mexanizmi).
"""

import chromadb
from ingest import ingest_all, Chunk
from embeddings import EmbeddingClient, embed_chunks

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "company_handbook"


def build_vector_store(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """
    Sənədləri oxuyur, chunk-lara bölür, embedding hesablayır və Chroma-da saxlayır.
    Chroma-nın PersistentClient-i istifadə olunur ki, məlumat disk-də qalsın
    (proqram bağlansa belə itməsin).
    """
    print("1) Sənədlər oxunur və chunk-lara bölünür...")
    chunks = ingest_all(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print(f"\n2) {len(chunks)} chunk üçün embedding hesablanır...")
    embedded = embed_chunks(chunks)

    print("\n3) Chroma vektor bazasına yazılır...")
    client = chromadb.PersistentClient(path=persist_dir)

    # Əgər kolleksiya artıq varsa, təmiz başlamaq üçün silib yenidən yaradırıq
    # (təkrar ingestion-da köhnə/dublikat məlumat qalmasın deyə)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    collection.add(
        ids=[f"{item['chunk'].source}_{item['chunk'].chunk_id}" for item in embedded],
        embeddings=[item["embedding"] for item in embedded],
        documents=[item["chunk"].text for item in embedded],
        metadatas=[
            {
                "source": item["chunk"].source,
                "chunk_id": item["chunk"].chunk_id,
                "start_char": item["chunk"].start_char,
                "end_char": item["chunk"].end_char,
            }
            for item in embedded
        ],
    )

    print(f"✅ {len(embedded)} chunk Chroma-ya yazıldı (qovluq: {persist_dir}/)")
    return collection


def search(
    query: str,
    top_k: int = 3,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_client: EmbeddingClient | None = None,
) -> list[dict]:
    """
    İstifadəçi sualına ən oxşar top_k chunk-ı tapır.

    Return: [{"text": str, "metadata": dict, "distance": float}, ...]
    (distance nə qədər kiçikdirsə, oxşarlıq bir o qədər yüksəkdir)
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)

    embedding_client = embedding_client or EmbeddingClient()
    query_embedding = embedding_client.embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return output


if __name__ == "__main__":
    print("=== VEKTOR BAZASININ QURULMASI ===\n")
    build_vector_store()

    print("\n\n=== OXŞARLIQ AXTARIŞI TESTİ ===")

    test_queries = [
        "Neçə gün illik ödənişli məzuniyyət haqqım var?",
        "Uzaqdan işləmək üçün nə etməliyəm?",
        "Şirkət nə vaxt təsis edilib?",
    ]

    for query in test_queries:
        print(f"\nSual: {query}")
        results = search(query, top_k=2)
        for rank, r in enumerate(results, start=1):
            print(f"  #{rank} (məsafə={r['distance']:.4f}, mənbə={r['metadata']['source']}, "
                  f"chunk={r['metadata']['chunk_id']}):")
            print(f"      {r['text'][:120]}...")
