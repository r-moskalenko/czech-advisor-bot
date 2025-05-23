import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-ada-002",
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef,
)

client = OpenAI(
    api_key=openai_api_key
)


def load_documents_from_directory(directory_path):
    """
    Load documents from a specified directory.
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r') as file:
                documents.append({"id": filename, "content": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    """
    Split text into chunks of specified size with overlap.
    """
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size - chunk_overlap)
    ]

directory_path = "./documents"
documents = load_documents_from_directory(directory_path)

print("===loading documents===")
print(f"loaded {len(documents)} documents from {directory_path}")

# Split documents into chunks

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["content"])
    print("===splitting document into chunks===")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_{i}",
            "content": chunk
        })

print(f"split into {len(chunked_documents)} chunks")

def get_openai_embedding(text):
    """
    Get OpenAI embedding for a given text.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    #print(response.data[0].embedding)
    embedding = response.data[0].embedding
    print("===Generating embeddings===")
    return embedding

for doc in chunked_documents:
    # Get the embedding for the chunk
    embedding = get_openai_embedding(doc["content"])
    # Add the chunk to the collection
    doc["embedding"] = embedding

print(doc["embedding"])

for doc in chunked_documents:
    # Add the chunk to the collection
    collection.add(
        documents=[doc["content"]],
        metadatas=[{"id": doc["id"]}],
        embeddings=[doc["embedding"]],
    )

print("===Adding documents to collection===")

def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)

    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

    print("=== Returning relevant chunks ===")
    return relevant_chunks

    for idx, document in enumerate(relevant_chunks):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
