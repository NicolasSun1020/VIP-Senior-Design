from transformers import pipeline

# Load BART model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text
input_text = (
    "Please summarize the following content into several key points:\n"
    "Deep learning is a subset of machine learning that focuses on utilizing neural networks to perform tasks such as classification, regression, and representation learning. "
    "The field takes inspiration from biological neuroscience and is centered around stacking artificial neurons into layers and training them to process data. "
    "The adjective deep refers to the use of multiple layers (ranging from three to several hundred or thousands) in the network. "
    "Methods used can be either supervised, semi-supervised or unsupervised. "
    "Some common deep learning network architectures include fully connected networks, deep belief networks, recurrent neural networks, convolutional neural networks, generative adversarial networks, transformers, and neural radiance fields. "
    "These architectures have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, "
    "where they have produced results comparable to and in some cases surpassing human expert performance. "
    "Early forms of neural networks were inspired by information processing and distributed communication nodes in biological systems, particularly the human brain. "
    "However, current neural networks do not intend to model the brain function of organisms, and are generally seen as low-quality models for that purpose."
)

# Run the summarization pipeline
summary = summarizer(input_text, min_length=60, max_length=180)[0]["summary_text"]

# Post-processing: split summary into bullet points by sentence
import re
sentences = re.split(r'(?<=[.!?]) +', summary)
bullet_points = "\n".join(f"- {s}" for s in sentences if len(s.strip()) > 0)

# Print result
print("🔍 Key Points Summary (Bullet Points):\n")
print(bullet_points)

# Save to file
with open("output.txt", "w") as f:
    f.write(bullet_points)

print("\n✅ Summary saved to: output.txt")



