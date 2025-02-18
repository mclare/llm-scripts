# Usage:
#   python prompt-and-record.py "Why is the sky blue?" --model llama2
#
# About:
#   Using my mostly Markdown notes, I want to generate a Brightspace quiz to study by completing practice questions. (Assessment for learning)
#   The Ollama AI model can generate quiz questions (multiple choice, and multi-select) based on the key terms and key points identified in the notes and will likely be run on a very low power device (Raspberry Pi5).
#   Because of the lower power device, the AI model will be run in a separate process and the results will be stored in a file for later use, adopting an Incremental Summarization approach https://aclanthology.org/2024.findings-emnlp.220.pdf
#   Finally, the quiz questions will be stored in a Brightspace CSV format for easy import into the LMS.
#   We need to be sure we're always working in UTF8 encoding to avoid any issues with special characters.


# To-do:
# 1. (done) Iterate through all markdown files in the folder. As needed, we'll make use of Incremental Summarization to generate both a key points summary and quiz questions
# 1.1 (done) Check the results directory contains and existing summary and quiz questions, if not, the next step is to generate them
# 2. (done) Use the Ollama AI model to identify the topic of the note, generate a summary of key point, and create a quiz questions (multiple choice and multiple select) for each key concept, product, service, resource, organizatiion, and term identified in the summary
# 2.1 (done) Summary can be generated using extractive summarization techniques and stored in the results directory as JSON
# 2.2 Quiz multiple choice questions can be generated using the key terms and key points identified in the summary and stored in Brightspace CSV format for either multiple choice or multiple select questions
# Each questions needs the following properties:
#   ID: Unique identifier for the question (e.g., identified topic + question number)
#   Title: Title of the question
#   QuestionText: The actual question text
#   Points: Points assigned to the question, should always be 1
#   Difficulty: Difficulty level, value between 1 and 10
#   Hint: Hint text
#   Feedback: Feedback text, which is not context-aware and will be the same for all answers and distracters, so it should focus on the concept being tested
#   
#   Parts of Brighspace CSV format we will not use:
#   Image: Optional image to include in the question, which this script will not use!
#   InitialText: Initial text for the response, which this script will not use!
#   AnswerKey: Answer key text, which this script will not use!
# 3. Use the Ollama AI model to generate a quiz question for each key term identified in the summary
# 3.1 TBD how to use the past summarization to generate the quiz questions. Options include: Literal concatenation, analysis of the summary, or a combination of both 🤷‍♂️

# Sample Brightspace CSV format:

"""
# Sample Brightspace CSV format:

//MULTIPLE CHOICE QUESTION TYPE,,,,
//Options must include text in column3,,,,
NewQuestion,MC,,,
ID,CHEM110-237,,,
Title,This is a multiple choice question,,,
QuestionText,This is the question text for MC1,,,
Points,1,,,
Difficulty,1,,,
Option,100,This is the correct answer,,This is feedback for option 1
Option,0,This is incorrect answer 1,,This is feedback for option 2
Option,0,This is incorrect answer 2,,This is feedback for option 3
Option,25,This is partially correct,,This is feedback for option 4
Hint,This is the hint text,,,
Feedback,This is the feedback text,,,
,,,,
,,,,
//TRUE / FALSE QUESTION TYPE,,,,
NewQuestion,TF,,,
ID,CHEM110-238,,,
Title,This is a True/False question,,,
QuestionText,This is the question text for TF1,,,
Points,1,,,
Difficulty,1,,,
TRUE,100,This is feedback for 'TRUE',,
FALSE,0,This is feedback for 'FALSE',,
Hint,This is the hint text,,,
Feedback,This is the feedback text,,,
,,,,
,,,,
//MULTISELECT QUESTION TYPE,,,,
//Options must include text in column3,,,,
NewQuestion,MS,,,
ID,CHEM110-239,,,
Title,This is a Multi-Select question,,,
QuestionText,This is the question text for MS1,,,
Points,10,,,
Difficulty,5,,,
Scoring,RightAnswers,,,
Option,1,This is option 1 text,,This is feedback for option 1
Option,0,This is option 2 text,,This is feedback for option 2
Option,1,This is option 3 text,,This is feedback for option 3
Hint,This is the hint text,,,
Feedback ,This is the feedback text,,,
"""
import os
import argparse
import json
import requests
import tiktoken
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkdownAnalyzer:
    def __init__(self, model: str, notes_dir: str, results_dir: str):
        self.model = model
        self.notes_dir = Path(notes_dir)
        self.results_dir = Path(results_dir)
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.token_limit = 2048
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_markdown_files(self) -> List[Path]:
        """Recursively find all markdown files in the given directory."""
        return list(self.notes_dir.rglob("*.md"))

    def read_markdown_file(self, filepath: Path) -> str:
        """Read a markdown file and return its contents as text."""
        try:
            return filepath.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return ""

    def check_existing_results(self, md_file: Path) -> bool:
        """Check if results already exist for the given markdown file."""
        base_name = md_file.stem
        summary_file = self.results_dir / f"{base_name}_summary.json"
        quiz_file = self.results_dir / f"{base_name}_quiz.json"
        concept_file = self.results_dir / f"{base_name}_concepts.json"
        return all(f.exists() for f in [summary_file, quiz_file, concept_file])

    def query_ollama(self, prompt: str, retries: int = 3) -> str:
        """Send a request to Ollama with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "prompt": prompt,
                        "model": self.model,
                        "stream": False
                    },
                    timeout=60 * 60
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to query Ollama after {retries} attempts")
                    return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into chunks that fit within token limits."""
        tokens = self.encoder.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(self.encoder.decode(chunk_tokens))
        return chunks

    def analyze_content(self, content: str) -> Tuple[Dict, List[Dict], List[Dict]]:
        """Generate summary, quiz questions, and extract key concepts."""
        # Process content in chunks
        chunks = self.split_text_into_chunks(content)
        summaries = []
        
        # Generate summaries for each chunk
        for chunk in chunks:
            summary_prompt = (
                "Analyze the following study notes and provide a detailed summary "
                "focusing on key points and main ideas:\n\n" + chunk
            )
            summary = self.query_ollama(summary_prompt)
            if summary:
                summaries.append(summary)

        full_summary = "\n".join(summaries)

        # Generate quiz questions
        quiz_prompt = (
            "Based on the content, generate 5 multiple choice questions. "
            "Format each question as a JSON object with 'question', 'options' (array), "
            "and 'correct_answer' fields."
        )
        quiz_response = self.query_ollama(quiz_prompt)
        
        try:
            quiz_questions = json.loads(quiz_response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse quiz questions as JSON, generating structured format")
            quiz_questions = [{"question": quiz_response, "options": [], "correct_answer": ""}]

        # Extract key concepts
        concept_prompt = (
            "Extract and list all key terms, concepts, products, services, and acronyms "
            "from the content. Format the response as a JSON array of objects with 'term' "
            "and 'category' fields."
        )
        concept_response = self.query_ollama(concept_prompt)
        
        try:
            concepts = json.loads(concept_response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse concepts as JSON, generating structured format")
            concepts = [{"term": concept_response, "category": "uncategorized"}]

        return {
            "summary": full_summary,
            "length": len(content),
            "chunks_processed": len(chunks)
        }, quiz_questions, concepts

    def process_files(self):
        """Process all markdown files and generate analysis results."""
        md_files = self.get_markdown_files()
        logger.info(f"Found {len(md_files)} markdown files to process")

        for md_file in md_files:
            if self.check_existing_results(md_file):
                logger.info(f"Skipping {md_file.name}, results already exist")
                continue

            logger.info(f"Processing {md_file.name}")
            content = self.read_markdown_file(md_file)
            
            if not content:
                continue

            summary_data, quiz_questions, concepts = self.analyze_content(content)
            
            # Save results
            base_name = md_file.stem
            
            # Save summary
            summary_file = self.results_dir / f"{base_name}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            # Save quiz questions
            quiz_file = self.results_dir / f"{base_name}_quiz.json"
            with open(quiz_file, 'w', encoding='utf-8') as f:
                json.dump(quiz_questions, f, indent=2, ensure_ascii=False)

            # Save concepts
            concept_file = self.results_dir / f"{base_name}_concepts.json"
            with open(concept_file, 'w', encoding='utf-8') as f:
                json.dump(concepts, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed {md_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze markdown study notes using Ollama AI")
    parser.add_argument("--model", default="dolphin-mistral", help="Ollama model to use")
    parser.add_argument("--notes-dir", default="./notes", help="Directory containing markdown notes")
    parser.add_argument("--results-dir", default="./results", help="Directory to store results")
    args = parser.parse_args()

    analyzer = MarkdownAnalyzer(args.model, args.notes_dir, args.results_dir)
    analyzer.process_files()

if __name__ == "__main__":
    main()
