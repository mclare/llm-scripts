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
# 1.1 Check the results directory contains and existing summary and quiz questions, if not, the next step is to generate them
# 2. Use the Ollama AI model to identify the topic of the note, generate a summary of key point, and create a quiz questions (multiple choice and multiple select) for each key concept, product, service, resource, organizatiion, and term identified in the summary
# 2.1 Summary can be generated using extractive summarization techniques and stored in the results directory as JSON
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
import markdown
import json
import requests
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Chat with Ollama AI and save responses.")
parser.add_argument("--model", default="dolphin-mistral", help="The AI model to use (default: dolphin-mistral).")
parser.add_argument("--notes-dir", dest="notes_dir", default="./notes", help="Location of study notes.")
parser.add_argument("--results-dir", dest="results_dir", default="./results", help="Directory to store results.")
args = parser.parse_args()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def get_markdown_files(notes_dir):
    """Recursively find all markdown files in the given directory."""
    return [f for f in Path(notes_dir).rglob("*.md")]

def read_markdown_file(filepath):
    """Read a markdown file and return its contents as text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def check_existing_results(results_dir, md_file):
    """Check if the results directory contains existing summary and quiz files for the given markdown file."""
    base_name = Path(md_file).stem  # Get the filename without extension
    summary_file = Path(results_dir) / f"{base_name}.json"
    quiz_file = Path(results_dir) / f"{base_name}.csv"
    return summary_file.exists() and quiz_file.exists()

def query_ollama(prompt):
    """Send a request to the Ollama AI model to generate responses."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": args.model, "prompt": prompt},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""

def generate_summary_and_questions(content):
    """Generate a summary and quiz questions using Ollama."""
    summary_prompt = f"Summarize the following study notes into key points:\n{content}"
    summary = query_ollama(summary_prompt)
    
    quiz_prompt = f"Based on the following summary, generate multiple choice and multi-select quiz questions:\n{summary}"
    quiz_questions = query_ollama(quiz_prompt)
    
    return summary, quiz_questions

def parse_markdown_notes(notes_dir, results_dir):
    """Process markdown notes, checking for existing results, and generating new ones if needed."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)  # Ensure results directory exists
    
    for md_file in get_markdown_files(notes_dir):
        if check_existing_results(results_dir, md_file):
            print(f"Skipping {md_file}, results already exist.")
            continue
        
        content = read_markdown_file(md_file)
        summary, quiz_questions = generate_summary_and_questions(content)
        
        base_name = Path(md_file).stem
        summary_file = Path(results_dir) / f"{base_name}.json"
        quiz_file = Path(results_dir) / f"{base_name}.csv"
        
        # Save summary as JSON
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, indent=4)
        
        # Save quiz questions as CSV
        with open(quiz_file, "w", encoding="utf-8") as f:
            f.write(quiz_questions)
        
        print(f"Generated summary and quiz for {md_file}")

# Example usage
if __name__ == "__main__":
    parse_markdown_notes(args.notes_dir, args.results_dir)
