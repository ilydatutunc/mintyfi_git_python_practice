from transformers import pipeline
import torch
import time

#device = -1 /cpu kullanimi icin
device = 0 if torch.cuda.is_available() else -1 #gpu kullanimi icin

summarizer = pipeline("summarization", device=device)

texts = [
"Hugging Face is creating a tool that democratizes AI. It provides access to pre-trained models for tasks like summarization, translation, classification, and question answering.",
"Transformers are state-of-the-art models that have revolutionized natural language processing. They enable machines to understand and generate human-like language.",
"The Eiffel Tower is one of the most famous landmarks in Paris. It was built in 1889 for the World's Fair and attracts millions of tourists every year.",
"Artificial intelligence (AI) is rapidly transforming many aspects of society, from healthcare and education to transportation and finance. One of the most significant developments in recent years has been the rise of large language models (LLMs) like GPT and BERT, which are capable of understanding and generating human-like text. These models are being used to build tools that can write essays, translate languages, answer questions, and even create poetry. However, the growing influence of AI also raises concerns about bias, misinformation, and the future of work. Policymakers, researchers, and technology companies are now working together to ensure that AI is developed and deployed responsibly. By promoting transparency, fairness, and accountability, it is possible to harness the benefits of AI while minimizing its risks."

]

results = []
durations = []

for text in texts:
    start = time.time()
    result = summarizer(text)
    end = time.time()

    results.append(result[0])
    durations.append(end - start)

for i in range(len(texts)):
    print(f"\nText {i+1}:")
    print(f"Input:\n{texts[i]}\n")
    print(f"Summary:\n{results[i]['summary_text']}\n")
    print(f"Duration: {durations[i]:.2f} seconds\n")
