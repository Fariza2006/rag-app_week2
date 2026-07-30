"""
rag_pipeline.py
-----------------
Checkpoint 4: Retrieval + Prompt Qurulması

Bu modul RAG-ın "beyni"dir:
1. İstifadəçi sualını alır
2. Vektor bazasından (Chroma) ən oxşar chunk-ları tapır (retrieval)
3. Çəkilmiş chunk-ları AYDIN KONTEKST/TƏLİMAT AYRIMI ilə prompt-a qurur
4. LLM-ə göndərir və cavabı qaytarır

VACİB PRİNSİP: Kontekst (sənəddən gələn məlumat) və təlimat (modelə necə davranmalı
olduğu göstərişi) prompt-da AYRI-AYRI, aydın işarələnmiş bölmələrdə olmalıdır ki,
model bunları qarışdırmasın (məsələn, sənəddəki mətni təlimat kimi qəbul etməsin).
"""

from vector_store import search
from llm_client import chat


SYSTEM_PROMPT = """Sən şirkət daxili sənədləri əsasında sual-cavab verən köməkçisən.

QAYDALAR:
1. Cavabını YALNIZ aşağıda "KONTEKST" bölməsində verilən mətnə əsaslandır.
2. Əgər kontekstdə sualın cavabı yoxdursa, uydurma — açıq şəkildə de ki, bu
   məlumat sənədlərdə yoxdur.
3. Cavabında hansı mənbədən (fayl adı və chunk nömrəsi) istifadə etdiyini qeyd et.
4. Qısa və dəqiq cavab ver, lazımsız təfərrüata getmə.
"""


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Çəkilmiş chunk-ları və sualı AYDIN AYRILMIŞ bölmələrlə user prompt-a çevirir.

    Struktur:
    - "### KONTEKST" bölməsi: hər chunk öz mənbəyi ilə etiketlənir
    - "### SUAL" bölməsi: istifadəçinin əsl sualı

    Bu ayrım vacibdir ki, model kontekst mətnini "təlimat" kimi qəbul etməsin
    (prompt injection-a bənzər qarışıqlığın qarşısını alır) və mənbəyə istinad edə bilsin.
    """
    context_blocks = []
    for i, item in enumerate(retrieved_chunks, start=1):
        meta = item["metadata"]
        context_blocks.append(
            f"[Mənbə {i}: {meta['source']}, chunk #{meta['chunk_id']}]\n{item['text']}"
        )

    context_section = "\n\n".join(context_blocks)

    prompt = f"""### KONTEKST (yalnız bu mətnə əsaslan)
{context_section}

### SUAL
{question}

### TƏLİMAT
Yuxarıdakı KONTEKST bölməsindəki məlumata əsasən SUAL-a cavab ver. Kontekstdə
cavab yoxdursa, bunu açıq şəkildə bildir. Hansı mənbədən istifadə etdiyini qeyd et."""

    return prompt


def answer_question(question: str, top_k: int = 3) -> dict:
    """
    Tam RAG pipeline: retrieval + prompt qurulması + LLM cavabı.

    Return:
        {
            "question": str,
            "answer": str,
            "retrieved_chunks": list[dict],  # istifadə olunan mənbələr
            "prompt_used": str,               # şəffaflıq üçün, hansı prompt göndərildi
        }
    """
    # 1) Retrieval
    retrieved_chunks = search(question, top_k=top_k)

    # 2) Prompt qurulması (aydın kontekst/təlimat ayrımı ilə)
    user_prompt = build_rag_prompt(question, retrieved_chunks)

    # 3) LLM-ə göndərmək
    answer = chat(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "prompt_used": user_prompt,
    }


if __name__ == "__main__":
    test_questions = [
        "Neçə gün illik ödənişli məzuniyyət haqqım var?",
        "Şirkət nə vaxt təsis edilib?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"SUAL: {q}")
        result = answer_question(q, top_k=2)
        print(f"\nCAVAB:\n{result['answer']}")
        print(f"\nİstifadə olunan mənbələr:")
        for item in result["retrieved_chunks"]:
            meta = item["metadata"]
            print(f"  - {meta['source']} (chunk #{meta['chunk_id']}, məsafə={item['distance']:.4f})")
