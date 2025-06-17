import os
import re
import json
import logging
import requests
import numpy as np
import tiktoken
from datetime import datetime
from rank_bm25 import BM25Okapi

OLLAMA_API_URL       = "http://localhost:xx"
CHAT_MODEL           = "gemma3:4b-it-q8_0"
DATA_DIR             = r"xxx"
HISTORY_DIR          = r"xxx"
MAIN_BM25_K          = 3
KW_BM25_K            = 2
MAX_KEYWORDS         = 10
MAX_CONTEXT_TOKENS   = 4000
ENCODING_NAME        = "cl100k_base"
TEMPERATURE          = 0.2
TOP_P                = 0.95
MAX_RESPONSE_TOKENS  = 2000

RESPONSE_INSTRUCTION = (
    "Yukarıdaki belgeler ile bu soruyu cevapla! "
    "Sonra hangi belgede bu cevabı bulduğunu kısaca belirt."
)

QUESTION_STOP = {
    "nedir","ne","kim","kimler","nerede","nerde",
    "nasıl","niçin","neden","hangi","kaç","ne zaman",
    "nerden","nereden","niye"
}

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_history_dir():
    if not os.path.isdir(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        logger.info(f"Created history directory: {HISTORY_DIR}")


def load_corpus_from_dir():
    docs, fnames = [], []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
                fnames.append(fname)
    logger.info(f"Loaded {len(docs)} documents from {DATA_DIR}")
    return docs, fnames


def build_bm25_index(docs):
    tokenized = [
        re.findall(r"[a-zçğıöşü]+", doc.lower())
        for doc in docs
    ]
    bm25 = BM25Okapi(tokenized)
    logger.info("Built BM25 index")
    return bm25


def call_llm_for_list(prompt_system: str, prompt_user: str) -> list[str]:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt_user}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 200
    }
    logger.debug("LLM list-call payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    r = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    logger.debug(f"LLM raw list response:\n{raw!r}")

    cleaned = re.sub(r"```(?:\w+)?\s*", "", raw).replace("```", "").strip()
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    arr_text = m.group(0) if m else cleaned

    try:
        arr = json.loads(arr_text)
        kws = [w.strip().lower() for w in arr if isinstance(w, str)]
        logger.debug(f"Parsed JSON list: {kws}")
        return kws
    except json.JSONDecodeError:
        logger.warning("JSON parse failed; falling back to regex")
        quotes = re.findall(r'"([^"]+)"', cleaned)
        if quotes:
            kws = [w.strip().lower() for w in quotes]
            logger.debug(f"Regex-quoted fallback: {kws}")
            return kws
        words = re.findall(r"[a-zçğıöşü]+(?: [a-zçğıöşü]+)*", cleaned.lower())
        logger.debug(f"Regex-word fallback: {words}")
        return words

def extract_keywords(query: str) -> list[str]:
    system = (
        "Sen bir Türkçe anahtar kelime çıkarma asistanısın. "
        "Bu soruya cevap vermek için hangi kelimeler aranmalı? "
        "Kesinlikle soru-ekleri (nedir, nasıl, kim, ne, hangi…) olmasın. "
        "Cevabını yalnızca JSON dizi formatında ver."
    )
    user = f"Soru: \"{query}\""
    kws = call_llm_for_list(system, user)
    filtered = [kw for kw in kws if kw not in QUESTION_STOP]
    trimmed = filtered[:MAX_KEYWORDS]
    logger.info(f"Extracted keywords (stop-filtered & trimmed): {trimmed}")
    return trimmed

def extract_additional_lists(query: str) -> dict:
    base_sys = "Sen bir Türkçe anahtar kelime çıkarma asistanısın. Çıktını JSON dizi formatında ver."
    prompts = {
        "subject":   f"Bu sorunun öznesi kim veya ne? max 3 Soru: \"{query}\"",
        "predicate": f"Bu sorunun yüklemi ne? max 2 Soru: \"{query}\"",
        "names":     f"Bu soruda özel isimler var mı? max 2 Soru: \"{query}\"",
        "multiword": f"Bu soruda çift kelimeler (örnek: haber kanalı, devlet işi) var mı? max 2 Soru: \"{query}\"",
        "typos": (
            "Yanlış yazılmış kelimeler veya kazayla ayrılmış birleşik kelimeler var mı? "
            "Düzeltmeleri yalnızca JSON dizi olarak ver. max 3"
            f"Soru: \"{query}\""
        ),
        "lemmas": (
            "Sen bir Türkçe kelime kök bulma asistanısın. "
            "Yalnızca kök hallerin listesini JSON dizi olarak ver (örn. [\"ağaç\",\"deniz\"]). "
            f"Soru: \"{query}\""
        )
    }
    result = {}
    for key, user_p in prompts.items():
        logger.debug(f"Extracting {key} with prompt: {user_p}")
        lst = call_llm_for_list(base_sys, user_p)
        lst = [w for w in lst if w not in QUESTION_STOP]
        result[key] = lst[:MAX_KEYWORDS]
        logger.info(f"{key.capitalize()} extracted: {result[key]}")
    return result

def retrieve_bm25(query, bm25, docs, fnames, k):
    tokens = re.findall(r"[a-zçğıöşü]+", query.lower())
    logger.debug(f"BM25 query tokens: {tokens}")
    scores = bm25.get_scores(tokens)
    idxs = np.argsort(scores)[::-1][:k]
    out = []
    for idx in idxs:
        sim = float(scores[idx])
        out.append({
            "text": docs[idx],
            "file_name": fnames[idx],
            "sim": sim,
            "type": "bm25"
        })
        logger.info(f"BM25: {fnames[idx]} score={sim:.4f}")
    return out


def chat_with_bm25(query, docs, fnames, bm25, all_kws):
    enc = tiktoken.get_encoding(ENCODING_NAME)
    main_ctx = retrieve_bm25(query, bm25, docs, fnames, MAIN_BM25_K)
    kw_ctx = []
    for kw in all_kws:
        kw_ctx.extend(retrieve_bm25(kw, bm25, docs, fnames, KW_BM25_K))
    seen, contexts = set(), []
    for c in main_ctx + kw_ctx:
        if c["file_name"] not in seen:
            contexts.append(c)
            seen.add(c["file_name"])
    logger.info(f"{len(contexts)} total contexts after deduplication")
    filtered = []
    filter_sys = (
        "Sen bir Türkçe soru-cevap asistanısın. "
        "Aşağıdaki belgeyle soruyu cevaplayabilir misin? 'Evet' veya 'Hayır' ile yanıtla."
    )
    for ctx in contexts:
        payload = {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": filter_sys},
                {"role": "user",   "content": f"Belge:\n{ctx['text']}\n\nSoru: \"{query}\""}
            ],
            "temperature": 0.0,
            "max_tokens": 3
        }
        r = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip().lower()
        if ans.startswith("evet"):
            filtered.append(ctx)
            logger.info(f"Kept {ctx['file_name']} (Evet)")
        else:
            logger.info(f"Dropped {ctx['file_name']} (Hayır)")
    if not filtered:
        logger.warning("All contexts dropped; falling back to top-5 by similarity")
        contexts.sort(key=lambda c: c["sim"], reverse=True)
        filtered = contexts[:5]
    sections = []
    for i, ctx in enumerate(filtered, 1):
        sections.append(f"{'*'*5} start of belge {i} {'*'*5}")
        sections.append(ctx["text"])
        sections.append(f"{'*'*5} end of belge {i} {'*'*5}")

    prompt = "\n\n".join(sections) + f"\n\nsoru: \"{query}\"\n" + RESPONSE_INSTRUCTION
    logger.debug(f"Full chat prompt:\n{prompt[:200]}…")
    while len(enc.encode(prompt)) > MAX_CONTEXT_TOKENS and filtered:
        filtered.pop()
        sec = []
        for j, c in enumerate(filtered, 1):
            sec.append(f"{'*'*5} start of belge {j} {'*'*5}")
            sec.append(c["text"])
            sec.append(f"{'*'*5} end of belge {j} {'*'*5}")
        prompt = "\n\n".join(sec) + f"\n\nsoru: \"{query}\"\n" + RESPONSE_INSTRUCTION

    # final LLM call
    final_payload = {
        "model":      CHAT_MODEL,
        "messages": [
            {"role": "system", "content": RESPONSE_INSTRUCTION},
            {"role": "user",   "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "top_p":       TOP_P,
        "max_tokens":  MAX_RESPONSE_TOKENS
    }
    logger.debug("Final chat payload…")
    res = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=final_payload)
    res.raise_for_status()
    answer = res.json()["choices"][0]["message"]["content"]
    return answer, final_payload, prompt

def main():
    ensure_history_dir()
    docs, fnames = load_corpus_from_dir()
    bm25 = build_bm25_index(docs)

    print("RAG ready.")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit","quit"):
            break

        try:
            main_kws = extract_keywords(query)
            logger.info(f"Main keywords: {main_kws}")

            extras = extract_additional_lists(query)
            all_kws = (
                main_kws
                + extras['subject'] + extras['predicate']
                + extras['names'] + extras['multiword']
                + extras['typos'] + extras['lemmas']
            )
            all_kws = list(dict.fromkeys([w.lower() for w in all_kws]))
            logger.info(f"All retrieval keywords ({len(all_kws)}): {all_kws}")
            answer, payload, prompt = chat_with_bm25(
                query, docs, fnames, bm25, all_kws
            )
            print("gem:", answer)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(HISTORY_DIR, f"{ts}.txt"), "w", encoding="utf-8") as hf:
                hf.write(f"Soru: {query}\n\nPrompt:\n{prompt}\n\nPayload:\n")
                hf.write(json.dumps(payload, ensure_ascii=False, indent=2))
                hf.write("\n\nCevap:\n" + answer)

        except Exception:
            logger.exception("Error in main loop")
            print("❗️ Something went wrong.")

if __name__ == "__main__":
    main()
