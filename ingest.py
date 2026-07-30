"""
ingest.py
----------
Checkpoint 1: Sənəd ingestion + chunking

Sənədləri oxuyur və məntiqli ölçü + overlap strategiyası ilə chunk-lara bölür.

NİYƏ OVERLAP VACİBDİR (chunk-sərhəd problemi):
Əgər sənəd sadə fixed-size (overlap-sız) üsulla bölünsə, vacib bir fakt/cümlə
tam olaraq iki chunk-ın sərhədində qala bilər — nəticədə RAG axtarışı bu
faktı TAM tapa bilmir, çünki nə əvvəlki, nə də sonrakı chunk cümlənin hamısını
əhatə etmir. Overlap (chunk-lar arasında təkrarlanan hissə) bu problemi aradan
qaldırır, çünki hər chunk özündən əvvəlki chunk-ın son N simvolunu da daxil edir.
"""

import os
import glob


class Chunk:
    """Bir mətn parçasını və onun mənbə metadata-sını saxlayan sadə struktur."""

    def __init__(self, text: str, source: str, chunk_id: int, start_char: int, end_char: int):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.start_char = start_char
        self.end_char = end_char

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(#{self.chunk_id}, {self.source}, {self.start_char}-{self.end_char}): '{preview}...'"


def load_documents(documents_dir: str) -> dict[str, str]:
    """
    documents_dir qovluğundakı bütün .txt fayllarını oxuyur.
    Return: {fayl_adı: mətn_məzmunu}
    """
    documents = {}
    for filepath in sorted(glob.glob(os.path.join(documents_dir, "*.txt"))):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            documents[filename] = f.read()
    return documents


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Mətni fixed-size + overlap strategiyası ilə chunk-lara bölür.

    Parametrlər:
        chunk_size: hər chunk-ın maksimum simvol sayı
        chunk_overlap: ardıcıl chunk-lar arasında təkrarlanan simvol sayı
                       (sərhəddə qalan faktları qorumaq üçün)

    Qeyd: chunk_overlap chunk_size-dan kiçik olmalıdır, əks halda sonsuz dövr yaranar.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap chunk_size-dan kiçik olmalıdır.")

    chunks = []
    start = 0
    chunk_id = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Sözün ortasında kəsilməməsi üçün, mümkünsə ən yaxın boşluğa qədər uzat/qısalt
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(Chunk(chunk_str, source, chunk_id, start, end))
            chunk_id += 1

        if end >= text_length:
            break

        # Növbəti chunk-ın başlanğıcı: overlap qədər geri çəkilir
        start = end - chunk_overlap

    return chunks


def ingest_all(
    documents_dir: str = "documents",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Bütün sənədləri yükləyir və chunk-lara bölür."""
    documents = load_documents(documents_dir)
    all_chunks = []

    for source, text in documents.items():
        chunks = chunk_text(text, source, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"'{source}': {len(text)} simvol -> {len(chunks)} chunk")

    return all_chunks


if __name__ == "__main__":
    print("=== NORMAL INGESTION (chunk_size=500, overlap=100) ===\n")
    chunks = ingest_all(chunk_size=500, chunk_overlap=100)
    for c in chunks:
        print(c)

    # ------------------------------------------------------------------
    # EDGE-CASE TEST: Chunk-sərhəd trick-i
    # Sənəddə "Standart illik ödənişli məzuniyyət müddəti 24 iş günüdür..."
    # cümləsi simvol 961-1060 aralığındadır. chunk_size=1000, overlap=0 ilə
    # bölsək, bu cümlə TAM olaraq iki chunk arasında bölünəcək.
    # ------------------------------------------------------------------
    print("\n\n=== EDGE-CASE TEST: Overlap OLMADAN (chunk_size=1000, overlap=0) ===")
    text = open("documents/company_handbook.txt", encoding="utf-8").read()
    no_overlap_chunks = chunk_text(text, "company_handbook.txt", chunk_size=1000, chunk_overlap=0)

    key_fact = "24 iş günüdür"
    key_sentence_start = "Standart illik ödənişli məzuniyyət"

    def _normalize(s: str) -> str:
        """Sətir keçidlərini boşluqla əvəz edir ki, yoxlama sətir sarılmasından təsirlənməsin."""
        return " ".join(s.split())

    found_complete = False
    for c in no_overlap_chunks:
        norm = _normalize(c.text)
        if key_fact in norm and key_sentence_start in norm:
            found_complete = True

    print(f"Açar fakt ('{key_fact}' + cümlənin əvvəli) BİR chunk-da tam var mı? "
          f"{'✅ BƏLİ' if found_complete else '❌ XEYR - cümlə chunk-lar arasında bölünüb!'}")
    for c in no_overlap_chunks:
        if key_fact in _normalize(c.text) or key_sentence_start in _normalize(c.text):
            print(f"  -> {c}")

    print("\n=== EDGE-CASE TEST: Overlap İLƏ (chunk_size=1000, overlap=200) ===")
    overlap_chunks = chunk_text(text, "company_handbook.txt", chunk_size=1000, chunk_overlap=200)

    found_complete_overlap = False
    for c in overlap_chunks:
        norm = _normalize(c.text)
        if key_fact in norm and key_sentence_start in norm:
            found_complete_overlap = True

    print(f"Açar fakt ('{key_fact}' + cümlənin əvvəli) BİR chunk-da tam var mı? "
          f"{'✅ BƏLİ - overlap problemi həll etdi' if found_complete_overlap else '❌ XEYR'}")
    for c in overlap_chunks:
        if key_fact in _normalize(c.text) or key_sentence_start in _normalize(c.text):
            print(f"  -> {c}")
