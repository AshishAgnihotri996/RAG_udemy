import asyncio
import os

from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):
        return False

from openai import AsyncOpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class ChatService:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Add it to your environment or .env file located in "
                f"{BASE_DIR}."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    async def ask(self, question: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning:free",
                messages=[{"role": "user", "content": question}],
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not reach OpenRouter. Check your internet connection and API configuration."
            ) from exc

        message = response.choices[0].message
        content = message.content if message is not None else ""
        return (content or "").strip()


async def main():
    chat_service = ChatService()
    question = "What is the capital of Maharashtra?"
    answer = await chat_service.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())