import anthropic
import faiss
import pickle
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

#KEYS
ANTHROPIC_API_KEY = "xx"
AZURE_OPENAI_API_KEY = "xx"
AZURE_OPENAI_ENDPOINT = "xx"

#Anthropic client
try:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logging.info("Anthropic client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Anthropics client: {e}")
    exit()

def embed_text_with_azure_openai(text, api_key, endpoint):
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "input": text,
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Embedding generated for text of length {len(text)}.")
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to generate embedding: {e}")
        raise

def load_faiss_index(index_path="faiss_index.bin", docs_path="documents.pkl"):
    try:
        faiss_index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            metadata = pickle.load(f)
        documents = metadata["documents"]
        file_names = metadata["file_names"]
        logging.info("FAISS index and documents loaded successfully.")
        return faiss_index, documents, file_names
    except Exception as e:
        logging.error(f"Failed to load FAISS index or documents: {e}")
        exit()

def search_faiss_index(query_embedding, faiss_index, documents, file_names,
                       top_k=10, distance_threshold=0.5):
    try:
        distances, indices = faiss_index.search(
            np.array([query_embedding], dtype=np.float32),
            top_k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "text": documents[idx],
                "file_name": file_names[idx],
                "distance": dist
            })

        if distance_threshold is not None:
            filtered = []
            for item in results:
                if item["distance"] < distance_threshold:
                    filtered.append(item)
            results = filtered
        logging.info(f"Found {len(results)} relevant documents.")
        return results
    except Exception as e:
        logging.error(f"Error searching FAISS index: {e}")
        return []

def parse_claude_response(claude_message):
    if not hasattr(claude_message, "content"):
        return "No content found in the response."
    output = ""
    for block in claude_message.content:
        if block.type == "text":
            output += block.text

            #citations append
            if getattr(block, "citations", None):
                citations_list = []
                for c in block.citations:
                    doc_title = c.get("document_title", "Unknown")
                    start_idx = c.get("start_char_index", 0)
                    end_idx = c.get("end_char_index", 0)
                    citations_list.append(f"[{doc_title} (chars {start_idx}-{end_idx})]")
                if citations_list:
                    output += " " + ", ".join(citations_list)

    return output.strip()


def chat_with_claude(query, relevant_documents, model="claude-3-5-sonnet-20241022", max_tokens=4000, temperature=0.2):
    system_prompt = (
        "Sen bir türkçe soru-cevap asistanısın. Sana yöneltilen sorulara yalnızca "
        "sağlanan belgelerden referans alarak cevap ver. Belgelerdeki metinleri "
        "değiştirmen gerekmez; ancak, varsa yazım ve dilbilgisi hatalarını düzelt. "
        "Ana fikir ve konudan sapmadığından emin ol ve cevaplarını her zaman "
        "dikkatlice değerlendir. Verdiğin her cevaba mutlaka belgelerden bir referans "
        "ekle. Sağlanan kaynaklarda bir bilgi bulunmaz ise: cevap verme, bulamadığını belirt. "
        "Referansları cevabın sonunda listele."
    )
    sources = [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": doc["text"]
            },
            "title": doc["file_name"],
            "context": "This is a trustworthy document.",
            "citations": {"enabled": True}
        }
        for doc in relevant_documents
    ]
    user_message = {
        "role": "user",
        "content": sources + [{"type": "text", "text": query}]
    }
    messages = [user_message]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        logging.debug(f"Raw response from Claude: {response}")
        return parse_claude_response(response)
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return "An unexpected error occurred. Please check the logs."


if __name__ == "__main__":
    index_path = "faiss_index.bin"
    docs_path = "documents.pkl"
    faiss_index, documents, file_names = load_faiss_index(index_path, docs_path)

    print("Welcome to the Claude Chat with Citations!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            query_embedding = embed_text_with_azure_openai(
                user_input,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_ENDPOINT
            )
        except Exception:
            print("Claude: Failed to generate query embedding. Please try again.")
            continue

        relevant_docs = search_faiss_index(query_embedding, faiss_index, documents, file_names, top_k=20, distance_threshold=0.5)
        if not relevant_docs:
            print("Claude: Sorry, I couldn't find any relevant documents.")
            continue

        response_text = chat_with_claude(user_input, relevant_docs)
        print(f"Claude: {response_text}")
