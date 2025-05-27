from vectordb import VectorDb

class AIAgent:
    def __init__(self, client, vector_store : VectorDb, embedding_model):
        self.client = client
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def generate(self, question : str) -> str:
        """
        Generate a response to a question using the vector store and OpenAI client.
        This method retrieves relevant chunks from the vector store based on the question,
        and then generates a response using the OpenAI client.
        """
        relevant_chunks = self.vector_store.query_documents(question)
        return self.generate_response(question, relevant_chunks)

    def generate_response(self, question, relevant_chunks):
        """ Generate a response using the OpenAI client based on the question and relevant chunks."""
        context = "\n\n".join(relevant_chunks)
        prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "Context:\n" + context + "\n\nQuestion:\n" + question
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]
        )

        answer = response.choices[0].message
        return answer
    
    def get_openai_embedding(self, text):
        """
        Get OpenAI embedding for a given text.
        """

        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        
        embedding = response.data[0].embedding
        print("===Generating embeddings===")
        return embedding
