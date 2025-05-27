import os
import logging
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils import embedding_functions
from aiogram import Bot, Dispatcher

from handlers import router
from vectordb.vectordb import VectorDb
from agent.agent import AIAgent
from preprocess.preprocessing import Preprocessor

async def main() -> None:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model,
    )

    client = OpenAI(
        api_key=openai_api_key
    )

    vector_store = VectorDb(
        db_type="chromadb",
        embedding_function=openai_ef,
    )
    
    agent = AIAgent(client, vector_store, embedding_model=embedding_model)

    preprocessor = Preprocessor(vector_store, agent, directory_path="./documents")

    # Configuration
    TOKEN = os.getenv("TG_TOKEN")

    # Bot initialization
    dp = Dispatcher(preprocessor=preprocessor, agent=agent)
    dp.include_router(router)
    bot = Bot(token=TOKEN)

    await dp.start_polling(bot)

# Start bot
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
