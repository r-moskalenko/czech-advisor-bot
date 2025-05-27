import os
from aiogram import F, Router, Bot
from aiogram.filters import CommandStart
from aiogram import types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message
from aiogram.filters import Command

from preprocess import Preprocessor

from dotenv import load_dotenv


router = Router()

# Handlers
@router.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.answer("""Welcome! To legal advizor using bot:
    1. Send a file for ingestion.

    2. Reply to that message and send: "/ingest" command.

    3. Ask any question that related to text in the document.""")

@router.message(Command("ingest"))
async def ingest_handler(message: types.Message, bot: Bot, preprocessor : Preprocessor) -> None:
    if message.reply_to_message:
        file_id = message.reply_to_message.document.file_id
    else:
        return
    file = await bot.get_file(file_id)
    file_path = file.file_path
    
    await bot.download_file(file_path, destination=f"./documents/{file.file_path.split('/')[-1]}")
    file_path = f"./documents/{file.file_path.split('/')[-1]}"
    
    print(f"Received file path: {file_path}")
    
    # Preprocess the document
    preprocessor.preprocess_single_document(file_path)

@router.message(Command("ask"))
async def ask_handler(message: Message, bot: Bot, preprocessor: Preprocessor) -> None:
    question = message.text.replace("/ask", "").strip()
    if not question:
        await message.answer("Please provide a question after the /ask command.")
        return

    # Generate a response using the agent
    response = preprocessor.agent.generate(question)
    
    await message.answer(response)