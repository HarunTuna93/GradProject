import os
import re
import json
import logging
import faiss
import pickle
import requests
import numpy as np
import tiktoken
from datetime import datetime

OLLAMA_API_URL       = "http://localhost:xxx"
EMBED_MODEL          = "mxbai-embed-large"
CHAT_MODEL           = "gemma3:4b-it-q8_0"
INDEX_PATH           = "xxx.bin"
META_PATH            = "documents.pkl"
HISTORY_DIR          = r"xxx"
MAIN_SEM_K           = 5
KW_SEM_K             = 2
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


def load_index_and_metadata():
    logger.debug(f"Loading FAISS index from '{INDEX_PATH}'")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    logger.info(f"Index vectors: {index.ntotal}, metadata docs: {len(meta['documents'])}")
    return index, meta["documents"], meta["file_names"]


def embed(text: str) -> np.ndarray:
    payload = {"model": EMBED_MODEL, "input": [text]}
    logger.debug(f"Embedding payload: {json.dumps(payload, ensure_ascii=False)}")
    resp = requests.post(f"{OLLAMA_API_URL}/v1/embeddings", json=payload)
    resp.raise_for_status()
    emb = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    logger.debug(f"Received embedding (dim={emb.shape[0]}) preview={emb[:5]}")
    return emb


def retrieve_semantic(query_emb, index, docs, fnames, k):
    D, I = index.search(query_emb.reshape(1, -1), k)
    out = []
    for dist, idx in zip(D[0], I[0]):
        sim = 1.0 / (1.0 + dist)
        out.append({
            "text": docs[idx],
            "file_name": fnames[idx],
            "sim": sim,
            "type": "semantic"
        })
        logger.info(f"Semantic: {fnames[idx]} dist={dist:.4f} sim={sim:.4f}")
    return out


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
    logger.debug("LLM list‐call payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    r = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    cleaned = re.sub(r"```(?:\w+)?\s*", "", raw).replace("```", "").strip()
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    arr_text = m.group(0) if m else cleaned
    try:
        arr = json.loads(arr_text)
        return [w.strip().lower() for w in arr if isinstance(w, str)]
    except json.JSONDecodeError:
        quotes = re.findall(r'"([^"]+)"', cleaned)
        if quotes:
            return [w.strip().lower() for w in quotes]
        return re.findall(r"[a-zçğıöşü]+(?: [a-zçğıöşü]+)*", cleaned.lower())


def extract_keywords(query: str) -> list[str]:
    system = (
        "Sen bir Türkçe anahtar kelime çıkarma asistanısın. "
        "Bu soruya cevap vermek için hangi kelimeler aranmalı? "
        "Kesinlikle soru-ekleri (nedir, nasıl, kim, ne, hangi vs) olmasın."
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
        "subject":   f"Bu sorunun öznesi kim veya ne? Soru: \"{query}\"",
        "predicate": f"Bu sorunun yüklemi ne? Soru: \"{query}\"",
        "names":     f"Bu soruda özel isimler var mı? Soru: \"{query}\"",
        "multiword": f"Bu soruda çift kelimeler (haber kanalı, devlet işi vs) var mı? Soru: \"{query}\"",
        "typos": (
            "Yanlış yazılmış kelimeler veya kazayla ayrılmış birleşik kelimeler var mı? "
            "Düzeltmeleri yalnızca JSON dizi olarak ver. "
            f"Soru: \"{query}\""
        ),
        "lemmas": (
            "Sen bir Türkçe kelime kök bulma asistanısın. "
            "Yalnızca çekimli hallerin köklerini JSON dizi formatında ver (örn. [\"ağaç\",\"deniz\"]). "
            f"Soru: \"{query}\""
        )
    }
    result = {}
    for key, user_p in prompts.items():
        lst = call_llm_for_list(base_sys, user_p)
        lst = [w for w in lst if w not in QUESTION_STOP]
        result[key] = lst[:MAX_KEYWORDS]
        logger.info(f"{key.capitalize()} extracted: {result[key]}")
    return result


def chat_with_semantic(query, index, docs, fnames, all_kws):
    enc = tiktoken.get_encoding(ENCODING_NAME)
    main_ctx = retrieve_semantic(embed(query), index, docs, fnames, MAIN_SEM_K)
    kw_ctx = []
    for kw in all_kws:
        kw_ctx.extend(retrieve_semantic(embed(kw), index, docs, fnames, KW_SEM_K))
    seen, contexts = set(), []
    for c in main_ctx + kw_ctx:
        if c["file_name"] not in seen:
            contexts.append(c)
            seen.add(c["file_name"])
    logger.info(f"{len(contexts)} contexts after deduplication")
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
            "max_tokens": 5
        }
        r = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip().lower()
        if ans.startswith("evet"):
            filtered.append(ctx)
    if not filtered:
        contexts.sort(key=lambda c: c["sim"], reverse=True)
        filtered = contexts[:5]
    sections = []
    for i, ctx in enumerate(filtered, 1):
        sections.append(f"{'*'*5} start of belge {i} {'*'*5}")
        sections.append(ctx["text"])
        sections.append(f"{'*'*5} end of belge {i} {'*'*5}")

    prompt = "\n\n".join(sections) + f"\n\nsoru: \"{query}\"\n" + RESPONSE_INSTRUCTION
    while len(enc.encode(prompt)) > MAX_CONTEXT_TOKENS and filtered:
        filtered.pop()
        sec = []
        for j, c in enumerate(filtered, 1):
            sec.append(f"{'*'*5} start of belge {j} {'*'*5}")
            sec.append(c["text"])
            sec.append(f"{'*'*5} end of belge {j} {'*'*5}")
        prompt = "\n\n".join(sec) + f"\n\nsoru: \"{query}\"\n" + RESPONSE_INSTRUCTION
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
    res = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=final_payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"], final_payload, prompt


def main():
    ensure_history_dir()
    index, docs, fnames = load_index_and_metadata()

    print("RAG ready.")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit","quit"):
            break
        try:
            main_kws = extract_keywords(query)
            extras   = extract_additional_lists(query)
            all_kws  = (
                main_kws
                + extras['subject'] + extras['predicate']
                + extras['names']   + extras['multiword']
                + extras['typos']   + extras['lemmas']
            )
            all_kws = list(dict.fromkeys([w.lower() for w in all_kws]))
            logger.info(f"All retrieval keywords ({len(all_kws)}): {all_kws}")
            answer, payload, prompt = chat_with_semantic(
                query, index, docs, fnames, all_kws
            )
            print("gem:", answer)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(HISTORY_DIR, f"{ts}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Soru: {query}\n\nPrompt:\n{prompt}\n\nPayload:\n")
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
                f.write("\n\nCevap:\n" + answer)

        except Exception:
            logger.exception("Error in main loop")
            print("❗️ Bir hata oluştu. Lütfen loglara bakın.")


if __name__ == "__main__":
    main()
