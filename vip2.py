from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline

model_name = "google/pegasus-xsum"
example_text = """
Deep learning is a subset of machine learning that focuses on utilizing neural networks to perform tasks such as classification, regression, and representation learning. The field takes inspiration from biological neuroscience and is centered around stacking artificial neurons into layers and training them to process data. The adjective deep refers to the use of multiple layers (ranging from three to several hundred or thousands) in the network. Methods used can be either supervised, semi-supervised or unsupervised. Some common deep learning network architectures include fully connected networks, deep belief networks, recurrent neural networks, convolutional neural networks, generative adversarial networks, transformers, and neural radiance fields. These architectures have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Early forms of neural networks were inspired by information processing and distributed communication nodes in biological systems, particularly the human brain. However, current neural networks do not intend to model the brain function of organisms, and are generally seen as low-quality models for that purpose.
"""


pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

tokens = pegasus_tokenizer(example_text, truncation=True, padding="longest", return_tensors="pt")

encoded_summary = pegasus_model.generate(**tokens)
decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)

summarizer = pipeline("summarization", model=pegasus_model, tokenizer=pegasus_tokenizer, framework="pt")

summary = summarizer(example_text, min_length=30, max_length=150)

print(summary[0]["summary_text"])


output_file = "output.txt"

output_data = summary[0]["summary_text"]


with open(output_file, "w") as file:
    file.write(output_data)

print(f"data is already in {output_file}")
