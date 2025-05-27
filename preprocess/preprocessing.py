import os
import re
from agent import AIAgent
from vectordb import VectorDb
from typing import List

class Preprocessor:

    def __init__(self, vector_store : VectorDb, agent : AIAgent, directory_path="./documents"):
        self.directory_path = directory_path
        self.documents = self.load_documents_from_directory(directory_path)
        self.vector_store = vector_store
        self.agent = agent
        print(f"Loaded {len(self.documents)} documents from {directory_path}")

    def preprocess_single_document(self, document_path):
        """
        Preprocess a single document by loading it and splitting it into chunks.
        """
        with open(document_path, "r") as file:
            content = file.read()
            
        content = self.clean_text(content)
        print("===Splitting document into articles===")
        document = self.split_law_document_into_articles(content)
        print("Document articles:", document.keys())
        
        chunks = []
        
        for chapter, articles in document.items():
            print(f"Chapter: {chapter}")
            
            for article in articles:
                print(f"Article: {article['title']} {article['body'][:50]}...")  # Print first 50 characters
                chunks.extend(self.split_text(article['body']))
        
        chunked_documents = [
            {"id": f"{os.path.basename(document_path)}_{i}", "content": chunk}
            for i, chunk in enumerate(chunks)
        ]
        
        for doc in chunked_documents:
            # Get the embedding for the chunk
            embedding = self.agent.get_openai_embedding(doc["content"])
            # Add the chunk to the collection
            doc["embedding"] = embedding
        
        self.vector_store.save_document_chunks(chunked_documents)

    def load_documents_from_directory(self, directory_path):
        """
        Load documents from a specified directory.
        """
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                with open(os.path.join(directory_path, filename), "r") as file:
                    documents.append({"id": filename, "content": file.read()})
        return documents


    def split_text(self, text, chunk_size=500, chunk_overlap=20) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        """
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size - chunk_overlap)
        ]

    def split_documents_into_chunks(self, documents, chunk_size=1000, chunk_overlap=20):
        """
        Split each document into chunks.
        """
        chunked_documents = []
        for doc in documents:
            chunks = self.split_text(doc["content"], chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunked_documents.append({"id": f"{doc['id']}_{i}", "content": chunk})
        return chunked_documents
    
    def clean_text(self, text):
        # Remove all content within curly brackets (including multiline)
        text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\(.*?\)', '', text, flags=re.DOTALL)

        # Replace multiple whitespace characters (including newlines) with a single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing spaces
        return text.strip()
    
    def split_law_document_into_articles(self, text):
         # Split into chapters based on "Розділ" followed by Roman numerals and uppercase words
        chapter_pattern = r'(Розділ\s+[IVXLCDM]+\s+[А-ЯІЄҐ\s]+)'
        chapter_splits = re.split(chapter_pattern, text)

        # Prepare dictionary to store results
        result = {}

        # The first element before any "Розділ..." is the preamble or title
        if chapter_splits[0].strip():
            result['Preamble'] = [{'title':'Preamble', 'body': chapter_splits[0].strip() }]

        # Process each chapter and its articles
        for i in range(1, len(chapter_splits), 2):
            chapter_title = chapter_splits[i].strip()
            chapter_content = chapter_splits[i + 1]

            # Split into articles within the chapter
            article_pattern = r'(Стаття\s+\d+\.\s*)'
            article_splits = re.split(article_pattern, chapter_content)

            # Combine article titles and content
            articles = []
            for j in range(1, len(article_splits), 2):
                article_title = article_splits[j].strip()
                article_body = article_splits[j + 1].strip()
                articles.append({'title': article_title, 'body': article_body})

            result[chapter_title] = articles

        return result
