# rag_retrieval.py
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import math

# ----------- CONFIG ----------- #
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K = 5
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2' # Example; customize as wrapped in infra
default_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
COLLECTION_NAME = "support_docs_v2"  # Use a new or reset collection for new chunking
# -------------------------------- #

def chunk_document(text: str, chunk_size:int = CHUNK_SIZE, overlap:int = CHUNK_OVERLAP) -> List[str]:
    """Split a large text into token-based (here: word) overlapping chunks."""
    import re
    # In production, use tokenizer for true tokens, here we use words as approximation.
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        if end == len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def process_and_add_documents(docs: List[Dict[str, Any]], chroma_client:chromadb.Client, collection_name:str=COLLECTION_NAME):
    """
    docs: List of dicts: {'id': str, 'text': str, 'category': str, 'priority': str/int, 'date': str}
    """
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_embedding_function, metadata={"hnsw:space": "cosine"})
    chunk_records = []
    chunk_ids = []
    metadatas = []
    count = 0
    for doc in docs:
        doc_chunks = chunk_document(doc['text'])
        for idx, chunk in enumerate(doc_chunks):
            record_id = f"{doc['id']}_chunk{idx}"
            meta = {
                "doc_id": doc['id'],
                "chunk_idx": idx,
                "category": doc['category'],
                "priority": doc['priority'],
                "date": doc['date']
            }
            chunk_ids.append(record_id)
            chunk_records.append(chunk)
            metadatas.append(meta)
            count += 1
    # Add to Chroma (auto-embeds as embedding_function is set)
    if chunk_records:
        collection.add(
            documents=chunk_records,
            metadatas=metadatas,
            ids=chunk_ids
        )
    print(f"Added {count} chunks to collection '{collection_name}'")
    return collection

def query_retrieve(query: str, chroma_client:chromadb.Client, collection_name:str=COLLECTION_NAME, top_k=TOP_K):
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_embedding_function)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"]
    )
    hits = []
    # results['documents'], [ [doc1, doc2,...] ]
    for idx in range(len(results['ids'][0])):
        hit = {
            "chunk_text": results['documents'][0][idx],
            "metadata": results['metadatas'][0][idx],
            "distance": results['distances'][0][idx],
            "id": results['ids'][0][idx]
        }
        hits.append(hit)
    return hits

def assemble_response(hits: List[Dict[str, Any]]) -> str:
    """
    Assemble a support response including the top-k chunks with citations.
    Non-redundant: Merge adjacent or identical doc/chunk hits.
    """
    # To avoid duplicate or overlapping context, keep a set of chunk ids or doc+chunk_idx
    seen = set()
    paragraphs = []
    for hit in hits:
        meta = hit['metadata']
        ref = f"[Category: {meta['category']} | Date: {meta['date']}]"
        # Only show unique doc+chunk
        key = (meta['doc_id'], meta['chunk_idx'])
        if key in seen:
            continue
        seen.add(key)
        para = f"{hit['chunk_text']}\n{ref}"
        paragraphs.append(para)
    response = "\n\n".join(paragraphs)
    return response

# ---------- Example Usage --------- #
if __name__ == "__main__":
    # Simulate a mini-corpus, as database/embedding pipeline is automated in prod
    sample_docs = [
        {
            "id": "doc1",
            "text": """
Customer can reset their password by clicking 'Forgot Password' on the login page. They'll receive an email with instructions. If the customer does not receive the email, check their spam folder, or contact support. Troubleshooting password issues may also involve checking account status and ensuring email is not blocked.
            """,
            "category": "authentication",
            "priority": "high",
            "date": "2023-01-15"
        },
        {
            "id": "doc2",
            "text": """
To update billing information, log into the dashboard and select 'Billing'. Accepted payment methods include Visa, Mastercard, and PayPal. Address inaccuracies can prevent payment. Support can update payment details upon user request.
            """,
            "category": "billing",
            "priority": "medium",
            "date": "2023-02-10"
        },
        {
            "id": "doc3",
            "text": """
Service outages are displayed on the status page. To receive updates, subscribe to email alerts. Major incidents will be communicated through the dashboard and email.
            """,
            "category": "service",
            "priority": "critical",
            "date": "2023-03-12"
        },
        {
            "id": "doc4",
            "text": """
Account deletion is permanent. Before deleting an account, download required data. Deletion prevents future access and cannot be reverted. Support may be able to assist only before the process is completed.
            """,
            "category": "account",
            "priority": "high",
            "date": "2023-04-25"
        }
    ]
    # Set up Chroma DB
    client = chromadb.Client()
    # Process and (re)populate the collection
    process_and_add_documents(sample_docs, client)
    
    # ---- Test queries ---- #
    sample_queries = [
        "How can a customer reset their password?",
        "How do I get notified about outages?",
        "How do I update my payment method?",
        "Is it possible to recover a deleted account?"
    ]
    for q in sample_queries:
        retrieved = query_retrieve(q, client)
        response = assemble_response(retrieved)
        print(f"\n=== Query: {q} ===\n{response}\n")

    # ---- Spot check: Recall@K ---- #
    # Gold annotations:
    gold = [
        "password", "outage", "billing", "delete"
    ]
    hit_count = 0
    total = 0
    for i, q in enumerate(sample_queries):
        hits = query_retrieve(q, client)
        combined = " ".join([hit['chunk_text'] for hit in hits]).lower()
        if gold[i] in combined:
            hit_count += 1
        total += 1
    recall_at_k = hit_count / total
    print(f"Recall@{TOP_K}: {recall_at_k:.2f}")
