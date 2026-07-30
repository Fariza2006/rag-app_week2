# RAG (Retrieval-Augmented Generation) — "Sənədlərinlə Danış"

Bu layihə istifadəçinin öz sənədləri (bu nümunədə: şirkət daxili siyasət sənədi) haqqında
sual verə bildiyi və yalnız həmin sənədlərin məzmununa əsaslanan cavab aldığı RAG
pipeline-ıdır.

**İstifadə olunan texnologiyalar:**
- LLM: Hugging Face router API (Həftə 1-dəki `hf_client.py`-in davamı)
- Vektor verilənlər bazası: **Chroma** (lokal, server tələb etmir)
- Embedding: Hugging Face Inference API

## Quraşdırma

```bash
pip install chromadb requests python-dotenv
```

API açarının konfiqurasiyası Həftə 1-dəki eynidir — `.env` faylında `HF_API_TOKEN` saxlanılır (bax `.env.example`).

---

# Checkpoint 1: Sənəd Ingestion + Chunking

`ingest.py` faylı `documents/` qovluğundakı `.txt` sənədlərini oxuyur və **fixed-size + overlap** strategiyası ilə chunk-lara bölür.

## Chunking strategiyası

- **`chunk_size`** (default 500 simvol) — hər chunk-ın maksimum ölçüsü.
- **`chunk_overlap`** (default 100 simvol) — ardıcıl chunk-lar arasında təkrarlanan hissə.
- Sözün ortasında kəsilməməsi üçün chunk sərhədi ən yaxın boşluğa qədər uzadılır/qısaldılır.

## Niyə overlap vacibdir? (chunk-sərhəd problemi)

Test sənədimizdə (`documents/company_handbook.txt`) qəsdən belə bir vəziyyət yaradılıb: **"Standart illik ödənişli məzuniyyət müddəti 24 iş günüdür"** cümləsi elə yerdədir ki, `chunk_size=1000, overlap=0` ilə bölündükdə bu cümlə **iki chunk arasında, hətta söz ortasında** kəsilir.

### Test nəticəsi (real, `python ingest.py` çıxışı)

**Overlap OLMADAN (chunk_size=1000, overlap=0):**
```
❌ XEYR - cümlə chunk-lar arasında bölünüb!
  -> Chunk(#1, company_handbook.txt, 984-1848): 'məzuniyyət müddəti 24 iş günüdür və bu müddət təqvim ili üzr...'
```
(Chunk #0 "...Standart illik ödənişli" sözü ilə bitir — "məzuniyyət" sözünün özü belə kəsilir!)

**Overlap İLƏ (chunk_size=1000, overlap=200):**
```
✅ BƏLİ - overlap problemi həll etdi
  -> Chunk(#1, company_handbook.txt, 784-1780): 'ilər üçün illik ödənişli məzuniyyət hüququ nəzərdə tutulmuşd...'
```

Bu, real RAG sistemlərində ən çox rast gəlinən problemlərdən biridir: **overlap-sız sadə chunking mühüm faktları itirə bilər**, çünki nə axtarış sistemi, nə də LLM natamam mətndən tam cavab tapa bilmir.

## İşlətmək

```bash
python ingest.py
```

## Fayl strukturu

```
rag-app/
├── documents/
│   └── company_handbook.txt   # Nümunə sənəd (chunk-sərhəd trick-i ilə)
├── ingest.py                   # Checkpoint 1: ingestion + chunking
├── embeddings.py                # Checkpoint 2: embedding generasiyası
├── .env.example
├── .gitignore
└── README.md
```

---

# Checkpoint 2: Chunk-lar üçün Embedding Generasiyası

`embeddings.py` hər chunk üçün Hugging Face-in pulsuz **feature-extraction** (embedding) API-si ilə vektor hesablayır.

## İstifadə olunan model və endpoint

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-ölçülü vektor, kiçik və sürətli)
- **Endpoint:** `https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction`

## Necə işləyir

1. `ingest.py`-dən gələn chunk-lar **batch-lərlə** (default 16-lıq qruplar) embedding API-sinə göndərilir — hər chunk-ı ayrı-ayrı göndərmək əvəzinə, sorğu sayını azaltmaq üçün.
2. Hər batch üçün müvəqqəti xətalarda (429/500-504, şəbəkə xətası) avtomatik retry edilir (Checkpoint 4-dəki eyni prinsip).
3. Nəticə: hər chunk-a uyğun `{"chunk": Chunk, "embedding": [float, ...]}` formatında siyahı.

## İşlətmək

```bash
python embeddings.py
```

## Nümunə çıxış (format)

```
Embedding hesablandı: 5/5 chunk

=== NƏTİCƏ ===
Chunk(#0, company_handbook.txt, 0-485): '...' -> vektor ölçüsü: 384, ilk 3 dəyər: [-0.023, 0.041, 0.008]
Chunk(#1, company_handbook.txt, 385-882): '...' -> vektor ölçüsü: 384, ilk 3 dəyər: [0.015, -0.032, 0.019]
...
```

---

# Checkpoint 3: Vektor Saxlama + Oxşarlıq Axtarışı

`vector_store.py` chunk-ları və onların embedding-lərini **Chroma**-da (lokal, server tələb etməyən vektor verilənlər bazası) saxlayır və istifadəçi sualına ən oxşar chunk-ları tapır.

## Necə işləyir

1. **`build_vector_store()`** — sənədləri oxuyur (`ingest.py`), chunk-lara bölür, embedding hesablayır (`embeddings.py`) və Chroma-nın `PersistentClient`-i ilə **disk-də** saxlayır (`chroma_db/` qovluğu — proqram bağlansa belə məlumat itmir).
2. **`search(query, top_k)`** — istifadəçi sualını embedding-ə çevirir, Chroma-nın daxili **cosine/L2 məsafə** axtarışı ilə ən oxşar `top_k` chunk-ı qaytarır.

## İşlətmək

```bash
python vector_store.py
```

## Nümunə nəticə (format)

```
Sual: Neçə gün illik ödənişli məzuniyyət haqqım var?
  #1 (məsafə=0.31, mənbə=company_handbook.txt, chunk=2):
      şçilər üçün illik ödənişli məzuniyyət hüququ nəzərdə tutulmuşdur...

Sual: Uzaqdan işləmək üçün nə etməliyəm?
  #1 (məsafə=0.42, mənbə=company_handbook.txt, chunk=1):
      : İŞ SAATLARI VƏ UZAQDAN İŞLƏMƏ  Standart iş saatları həftə...
```

Nəticələr göstərir ki, hər sual **mövzuca ən uyğun** chunk-ı tapır (aşağı məsafə = yüksək oxşarlıq), yəni embedding + Chroma axtarışı düzgün işləyir.

## Qeyd

`chroma_db/` qovluğu `.gitignore`-dadır — bu, avtomatik yaranan verilənlər bazası faylıdır, GitHub-a yüklənmir (hər kəs öz kompüterində `python vector_store.py` işlədərək yenidən yarada bilər).
