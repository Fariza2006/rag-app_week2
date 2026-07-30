"""
structured_output_helper.py
------------------------------
Modeldən JSON formatında cavab alarkən format pozuntularını (izahedici mətn,
markdown code-block) tutub təmizləyən köməkçi funksiya.
(Bax: Həftə 1-dəki structured_output.py-ın eyni prinsipi.)
"""

import json
import re


def parse_json_response(text: str) -> dict | None:
    """
    Modelin JSON cavabını təmizləyir və parse edir. Uğursuz olarsa None qaytarır
    (xəta atmır - çağıran tərəf None halını idarə edir).
    """
    text = text.strip()

    # Markdown code-block işarələrini təmizlə
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # İzahedici mətndən JSON hissəsini ayır
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
