import os
from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram import types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message
from aiogram.filters import Command
from dotenv import load_dotenv

from summarize import generate_summary

router = Router()

load_dotenv()
whitelist_env = os.getenv("WHILE_LIST")

if whitelist_env is None:
    raise ValueError("WHITE_LIST environment variable is not set.")

WHITELIST = set(map(int, whitelist_env.split(",")))

# Handlers
@router.message(CommandStart())
async def send_welcome(message: types.Message):
    if message.from_user.id not in WHITELIST:
        print(message.from_user.id)
        await message.answer("Access Denied.")
        return
    await message.answer("""Welcome! To summarize text using bot:
    1. Send a text to summarize.

    2. Reply to that message and send: "/summarize"

    3. You will get response with summary from the bot.""")

@router.message(Command("summarize"))
async def summarize_handler(message: types.Message) -> None:
    if message.from_user.id not in WHITELIST:
        await message.answer("Access Denied.")
        return
    replied_text = message.reply_to_message.text
    if not replied_text:
        await message.answer("The message you're replying to doesn't contain any text.")
        return
    
    generated_summary = generate_summary(replied_text)
    await message.answer(f"Summary: {generated_summary}")
