from transformers import pipeline
import re

# Load the BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Read input text from file
with open("input.txt", "r", encoding="utf-8") as f:
    input_text = f.read()

# split the input into chunks
def chunk_text(text, max_words=700):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = chunk_text(input_text)

# Summarize each chunk
summaries = []
for i, chunk in enumerate(chunks):
    summary = summarizer(chunk, min_length=60, max_length=180)[0]["summary_text"]
    summaries.append(summary)

# Combine summaries
combined_summary = " ".join(summaries)

# Format into bullet points
sentences = re.split(r'(?<=[.!?]) +', combined_summary)
bullet_points = "\n".join(f"- {s}" for s in sentences if len(s.strip()) > 0)

# Print summary
print("🔍 Key Points Summary (Bullet Points):\n")
print(bullet_points)

# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(bullet_points)

print("\n✅ Summary saved to: output.txt")
