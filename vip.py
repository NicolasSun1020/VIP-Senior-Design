from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.generators.ollama import OllamaGenerator

import re

document = []
with open("info.txt", "r") as file:
    for line in file:
        document.append(Document(content=line.strip()))

document_store = InMemoryDocumentStore()
document_store.write_documents(document)

# document_store.write_documents(
# document.extend(
#     [
#         # Document(content = np.loadtxt("info.txt")),
#         Document(content="Super Mario was a famous basketball player"),
#         Document(content="Mario wins several championships and uses the fame to influence his home town"),
#         Document(
#             content="Super Mario was a successful basketball player, and play against his "
#             "biggest rival - Lebron several times in the championship games"
#         ),
#         Document(content="Mario won 5 championships, 3 MVPs, 5 FMVPs, 1 DPOY, 1 ROY"),
#         Document(content="Mario was chose to be in All NBA First Team 10 times, and All Defensive First Team 8 times"),
#     ]
# )



template = """
Given only the following information, create a similar quiz based on the content.
Ensure the question is clear and directly relevant to the provided context.
Ignore your own knowledge.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}
"""

pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OllamaGenerator(model="llama3.1", url="http://localhost:11434"))
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")
query = "create a similar questions in question 1"

response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})
output = response["llm"]["replies"]
cleaned_output = " ".join(output).replace("\n", "\n")

print(cleaned_output)