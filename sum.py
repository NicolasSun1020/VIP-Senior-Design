from transformers import pipeline, AutoTokenizer
import re

# Load the BART model and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Read input text from file
with open("input.txt", "r", encoding="utf-8") as f:
    input_text = f.read()

# Split text into chunks based on token limit
def chunk_text_by_tokens(text, max_tokens=1024):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
        start = end
    return chunks

chunks = chunk_text_by_tokens(input_text, max_tokens=900)

# Summarize each chunk
summaries = []
for i, chunk in enumerate(chunks):
    try:
        summary = summarizer(chunk, min_length=60, max_length=180)[0]["summary_text"]
        summaries.append(summary)
    except Exception as e:
        print(f"Error summarizing chunk {i}: {e}")

# Combine summaries
combined_summary = " ".join(summaries)

# Format into bullet points
sentences = re.split(r'(?<=[.!?]) +', combined_summary)
bullet_points = "\n".join(f"- {s.strip()}" for s in sentences if len(s.strip()) > 0)

# Print summary
print("Key Points Summary:\n")
print(bullet_points)

# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(bullet_points)

print("\n Summary saved to: output.txt")
