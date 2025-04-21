from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
import re

# Load Reference Content
def load_reference_context(filepath="input.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        return Document(content=file.read().strip())

# Load Questions with Answers
def load_questions(filepath="output_question.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read().strip()

    question_blocks = re.split(r'\n(?=What|Which|How)', content)
    documents = []

    for block in question_blocks:
        if "answer:" in block.lower():
            block = block.strip()
            documents.append(Document(content=block))

    return documents

# Prompt Template
template = """
You are given some context and a multiple-choice question created from it.

Context:
{{ context }}

Question:
{{ query }}

Evaluate the question using the following criteria (score from 1 to 5 each):
1. Relevance – Is it clearly derived from the the refrence context?
2. Clarity – Is the question phrased in a clear and unambiguous way?
3. Cognitive Depth – Does it go beyond simple recall?
4. Distractor Quality – Are the incorrect options plausible?
5. Answer Key Validity – Is the answer clearly correct based on the context?

Return your answer in this format:
{
  "Relevance": int,
  "Clarity": int,
  "Cognitive Depth": int,
  "Distractor Quality": int,
  "Answer Key Validity": int,
}
"""

# Setup Pipeline
pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OpenAIGenerator(
    api_key=Secret.from_token(""),
    model="gpt-4o-mini"
))
pipe.connect("prompt_builder", "llm")

# Evaluate the questions
def evaluate_questions(question_docs, reference_doc):
    with open("evaluated_questions.txt", "w", encoding="utf-8") as txt_file:
        for idx, question_doc in enumerate(question_docs):
            print(f"\nEvaluating Question {idx + 1}...\n{'='*40}")
            try:
                res = pipe.run({
                    "prompt_builder": {
                        "query": question_doc.content,
                        "context": reference_doc.content
                    }
                })
                output = res["llm"]["replies"][0]
                full_output = f"Question {idx + 1}:\n{question_doc.content}\n\nEvaluation:\n{output}\n{'-'*60}\n"
                print(full_output)
                txt_file.write(full_output)
            except Exception as e:
                error_msg = f"Error evaluating question {idx + 1}: {e}\n"
                print(error_msg)
                txt_file.write(error_msg)

if __name__ == "__main__":
    questions = load_questions("output_question.txt")
    context = load_reference_context("input.txt")
    evaluate_questions(questions, context)
