import os
import faiss
import numpy as np
import requests
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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


def load_documents_from_folder(folder_path):
    documents = []
    file_names = []
    logging.info(f"Loading documents from folder: {folder_path}")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    documents.append(file.read())
                    file_names.append(file_name)
                    logging.info(f"Loaded file: {file_name}")
            except Exception as e:
                logging.error(f"Failed to read file {file_name}: {e}")
    logging.info(f"Loaded {len(documents)} documents from folder.")
    return documents, file_names


def create_faiss_index(folder_path, api_key, endpoint, output_index_path="faiss_index.bin", output_docs_path="documents.pkl"):
    documents, file_names = load_documents_from_folder(folder_path)
    if not documents:
        logging.error("No documents found in the specified folder.")
        return
    embeddings = []
    for idx, doc in enumerate(documents):
        logging.info(f"Processing document {idx + 1}/{len(documents)}: {file_names[idx]}")
        try:
            embedding = embed_text_with_azure_openai(doc, api_key, endpoint)
            embeddings.append(embedding)
            logging.debug(f"Embedding for document {file_names[idx]}: {embedding[:5]}...")
        except Exception as e:
            logging.error(f"Skipping document {file_names[idx]} due to error: {e}")
            continue
    if not embeddings:
        logging.error("No embeddings were generated. Exiting.")
        return
    embeddings = np.array(embeddings, dtype=np.float32)
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embeddings)
    logging.info(f"FAISS index created with {len(embeddings)} embeddings.")
    try:
        faiss.write_index(faiss_index, output_index_path)
        with open(output_docs_path, "wb") as f:
            pickle.dump({"documents": documents, "file_names": file_names}, f)
        logging.info(f"FAISS index saved to '{output_index_path}'.")
        logging.info(f"Document metadata saved to '{output_docs_path}'.")
    except Exception as e:
        logging.error(f"Failed to save FAISS index or metadata: {e}")


if __name__ == "__main__":
    api_key = "xx"
    endpoint = "xx"
    folder_path = r"xx"
    output_index_path = "faiss_index.bin"
    output_docs_path = "documents.pkl"
    create_faiss_index(folder_path, api_key, endpoint, output_index_path, output_docs_path)
