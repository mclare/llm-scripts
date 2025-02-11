
import os
import asyncio
from ollama import AsyncClient

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='dolphin-mistral', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())