# Usage:
#   python prompt-and-record.py "Why is the sky blue?" --model llama2
#   or
#   python prompt-and-record.py --prompt-file input.md --model llama2
#   or
#   python prompt-and-record.py "Why is the sky blue?" --model llama2 --results-dir /config/results --results-file "sky-blue.txt"

import os
import argparse
import asyncio
import datetime
from ollama import AsyncClient

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Chat with Ollama AI and save responses.")
parser.add_argument("prompt", nargs="?", help="The prompt to send to the AI model.")
parser.add_argument("--prompt-file", dest="prompt_file", help="Path to a Markdown file containing the prompt.")
parser.add_argument("--model", default="dolphin-mistral", help="The AI model to use (default: dolphin-mistral).")
parser.add_argument("--results-dir", dest="results_dir", default="/config/results", help="Directory to store results.")
parser.add_argument("--results-file", dest="results_file", default="", help="Filename to store results (default: timestamped).")
args = parser.parse_args()

# Set Ollama host from environment variable or default to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Determine the prompt source (priority: command-line > markdown file > default)
if args.prompt:
    prompt = args.prompt
elif args.prompt_file:
    if not os.path.isfile(args.prompt_file):
        print(f"Error: Specified prompt file '{args.prompt_file}' does not exist.")
        exit(1)
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
else:
    prompt = f"Please tell me a joke for {datetime.datetime.now().strftime('The Year %Y, %B %d at %H:%M:%S')}"

# Ensure the results directory exists
os.makedirs(args.results_dir, exist_ok=True)

# Generate a timestamped filename if results_file is empty
if not args.results_file:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = os.path.join(args.results_dir, f"chat_{timestamp}.md")
else:
    OUTPUT_FILE = os.path.join(args.results_dir, args.results_file)

async def chat():
    message = {'role': 'user', 'content': prompt}
    
    client = AsyncClient(host=OLLAMA_HOST)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        async for part in await client.chat(args.model, messages=[message], stream=True):
            content = part['message']['content']
            print(content, end='', flush=True)  # Stream to console
            f.write(content)  # Write to file
            f.flush()  # Ensure immediate write to disk

asyncio.run(chat())

print(f"\nOutput saved to {OUTPUT_FILE}")