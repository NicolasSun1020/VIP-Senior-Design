from haystack import Document, Pipeline
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators import OpenAIGenerator

# Load and clean the document
documents = []
with open("output.txt", "r", encoding="utf-8") as file:
    for line in file:
        clean_line = line.strip()
        if clean_line.startswith("-"):
            documents.append(Document(content=clean_line[1:].strip()))

# Prompt template
template = """
Given only the following information, create a quiz with multiple choice question based on the content and the provided example questions.
Ensure the question is clear and relevant to the provided context.
Ignore your own knowledge.

Template:
What is mechanism of action for cocaine to increase DA signaling in the NAc?
A. Enhanced DAT activity on presynaptic terminals of NAc neurons.
B. Blocking DAT on presynaptic terminals of NAc neurons.
C. Enhanced VMAT activity in NAc neurons.
D. Blocking DAT on the presynaptic terminals of VTA neurons.
answer: B) Blocking DAT on presynaptic terminals of NAc neurons.

D-ephedrin can be reduced into METH by removing the __________ group.
A. beta-hydroxyl
B. alpha-methyl
C. NHCH3
D. all of the above
answer A) beta-hydroxyl


Context:
- {{ document.content }}

Question: {{ query }}
"""

# Set up the pipeline
pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_token("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"), model="gpt-4o-mini"))
pipe.connect("prompt_builder", "llm")

query = "Create questions based on the bullet point below."

# Run the pipeline for each document
for i, doc in enumerate(documents, start=1):
    response = pipe.run({
        "prompt_builder": {
            "query": query,
            "document": doc
        }
    })
    output = response["llm"]["replies"]
    cleaned_output = " ".join(output).replace("\n", "\n")
    print(cleaned_output)
    
