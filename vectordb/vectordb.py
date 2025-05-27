import chromadb
from chromadb.utils import embedding_functions

class VectorDb:
    def __init__(self, db_type: str, embedding_function, **kwargs):
        self.db_type = db_type
        self.config = kwargs
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = "document_qa_collection"
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    
    def query_documents(question, collection, n_results=2):
        results = collection.query(query_texts=question, n_results=n_results)

        relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

        print("=== Returning relevant chunks ===")
        return relevant_chunks

        # for idx, document in enumerate(relevant_chunks):
        #     doc_id = results["ids"][0][idx]
        #     distance = results["distances"][0][idx]
            
    def save_document_chunks(self, chunked_documents):
        """
        Save document chunks to the vector database collection.
        """
        for doc in chunked_documents:
            # Add the chunk to the collection
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{"id": doc["id"]}],
                embeddings=[doc["embedding"]],
            )

        print("===Adding documents to collection===")
