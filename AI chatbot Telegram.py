
import os
import logging
import asyncio
from collections import defaultdict, deque

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai

# Load secret keys from .env
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError(
        "Missing BOT_TOKEN or GEMINI_API_KEY in your .env file."
    )

ai_client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Keep a short conversation history for each Telegram user
user_histories = defaultdict(lambda: deque(maxlen=12))

SYSTEM_PROMPT = """
You are a helpful AI assistant inside Telegram.
Answer clearly and politely.
Help with general questions, learning, Python and other code,
summaries, translations, resumes, interview preparation, and emails.
If a question is unclear, ask for clarification.
Do not claim that you performed actions you did not perform.
"""


async def ask_gemini(user_id: int, message: str) -> str:
    history = user_histories[user_id]

    conversation = SYSTEM_PROMPT + "\n\n"

    for user_message, assistant_message in history:
        conversation += f"User: {user_message}\n"
        conversation += f"Assistant: {assistant_message}\n"

    conversation += f"User: {message}\nAssistant:"

    response = await ai_client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=conversation,
    )

    answer = response.text
    if not answer:
        answer = "Sorry, I couldn't generate a response. Please try again."

    history.append((message, answer))
    return answer


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "Hello! 👋 I'm your Telegram AI Assistant.\n\n"
        "You can ask me questions or use these commands:\n\n"
        "/help - Show available commands\n"
        "/summarize - Summarize text\n"
        "/translate - Translate text\n"
        "/code - Explain code\n"
        "/resume - Review resume text\n"
        "/interview - Generate interview questions\n"
        "/email - Draft an email\n"
        "/clear - Clear conversation history\n\n"
        "Just send me a message to start chatting!"
    )
    await update.effective_message.reply_text(welcome)


async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    await update.effective_message.reply_text(
        "How to use me:\n\n"
        "Send any question for an AI response.\n\n"
        "Examples:\n"
        "/summarize Explain this topic in simple words: machine learning\n"
        "/translate Translate 'Good morning' into Telugu\n"
        "/code Explain this Python code: print('Hello')\n"
        "/resume Review my resume: [paste resume text]\n"
        "/interview Give me 10 Python interview questions\n"
        "/email Write a job application email for a fresher\n"
        "/clear Clear your recent conversation history"
    )


async def clear_history(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id
    user_histories[user_id].clear()
    await update.effective_message.reply_text(
        "Your conversation history has been cleared."
    )


async def ai_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id
    message = " ".join(context.args).strip()

    if not message:
        await update.effective_message.reply_text(
            "Please add text after the command.\n"
            "Example: /summarize Artificial intelligence is..."
        )
        return

    prompts = {
        "summarize": "Summarize the following text clearly:\n",
        "translate": "Translate the following text as requested:\n",
        "code": "Explain the following code for a beginner:\n",
        "resume": "Review this resume and suggest improvements:\n",
        "interview": "Generate useful interview questions and answers:\n",
        "email": "Draft a professional email based on this request:\n",
    }

    command = context.command
    prompt = prompts.get(command, "") + message

    await generate_and_reply(update, user_id, prompt)


async def generate_and_reply(
    update: Update, user_id: int, message: str
):
    msg = update.effective_message
    await msg.reply_text("Thinking... 🤖")

    try:
        answer = await ask_gemini(user_id, message)

        # Telegram messages have a maximum length.
        for start_index in range(0, len(answer), 4000):
            await msg.reply_text(answer[start_index:start_index + 4000])

    except Exception:
        logger.exception("Gemini request failed")
        await msg.reply_text(
            "Sorry, I couldn't process that request. "
            "Please try again later or check the API configuration."
        )


async def handle_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id
    message = update.effective_message.text.strip()

    if not message:
        return

    await generate_and_reply(update, user_id, message)


def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history))

    for command in (
        "summarize",
        "translate",
        "code",
        "resume",
        "interview",
        "email",
    ):
        application.add_handler(
            CommandHandler(command, ai_command)
        )

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    print("Telegram AI Assistant is running!")
    application.run_polling()


if __name__ == "__main__":
    main()
