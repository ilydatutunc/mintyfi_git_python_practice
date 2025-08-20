import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def load_or_create_cache(pdf_path, embed_model, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    chunks_cache = os.path.join(cache_dir, "nutuk_chunks.pkl")
    embeddings_cache = os.path.join(cache_dir, "nutuk_embeddings.pkl")

    if os.path.exists(chunks_cache) and os.path.exists(embeddings_cache):
        print("Cache'den yükleniyor...")
        with open(chunks_cache, "rb") as f:
            chunks = pickle.load(f)
        with open(embeddings_cache, "rb") as f:
            embeddings = pickle.load(f)
        print(f"{len(chunks)} chunk ve embedding'ler cache'den yüklendi!")
        return chunks, embeddings

    print("Cache oluşturuluyor...")
    text = pdf_to_text(pdf_path)
    chunks = split_text(text)
    print(f"Toplam {len(chunks)} chunk oluşturuldu.")

    print("Embedding modeli ile embedding'ler oluşturuluyor...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    with open(chunks_cache, "wb") as f:
        pickle.dump(chunks, f)
    with open(embeddings_cache, "wb") as f:
        pickle.dump(embeddings, f)

    print("Cache oluşturuldu ve kaydedildi!")
    return chunks, embeddings

def find_relevant_chunks(query, chunks, embeddings, embed_model, top_k=12):
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def extract_date(answer):
    patterns = [
        r"\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}\b",
        r"\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'de\b",
        r"\b\d{4}\s+yılında\b",
        r"\b\d{4}\s+yılı\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(0)
    return "Cevap bulunamadı"

def ask_gemini(query, context, is_date_question=False, temperature=0.7, top_p=0.9, top_k=50):
    try:
        if not is_date_question:
            is_date_question = any(word in query.lower() for word in [
                'ne zaman', 'tarih', 'yıl', 'yılında', 'kaçıncı', 'hangi tarih'
            ])

        if is_date_question:
            prompt = f"""
Sen bir metin analiz aracısın. Görevin sadece metinde geçen SORU’ya ait TARİH bilgisini çıkarmaktır. 
Hiçbir yorum, tahmin veya ek bilgi verme. 
Metinde tarih yoksa sadece "Cevap bulunamadı" yaz. 
YANIT: Sadece tarih veya yıl formatında yaz (ör: 1 Kasım 1922 veya 1923).

METİN:
{context}

SORU:
{query}

YANIT:"""
        else:
            prompt = f"""
Sen bir metin analiz aracısın. Görevin metinden cevabı çıkarmak veya metinde cevap yoksa mantıklı bir çıkarım yapmaktır.
- Gereksiz yorum veya ek bilgi verme.
- Mantıklı çıkarım yapabilirsin.

METİN:
{context}

SORU:
{query}

YANIT:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        gen_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)
        response = model.generate_content(prompt, generation_config=gen_config)
        answer = response.text.strip()

        if is_date_question:
            answer = extract_date(answer)

        if not answer:
            answer = "Cevap bulunamadı."

        return answer

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def load_models():
    print("Embedding modeli yükleniyor (CPU)...")
    embed_model = SentenceTransformer("HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5", device="cpu")
    return embed_model

# ana pipeline
def main():
    pdf_path = "data/nutuk.pdf"
    embed_model = load_models()
    chunks, embeddings = load_or_create_cache(pdf_path, embed_model)

    while True:
        query = input("\nSorunuzu yazın (çıkmak için 'quit'): ")
        if query.lower() == 'quit':
            break

        relevant_chunks = find_relevant_chunks(query, chunks, embeddings, embed_model)
        context = "...".join(relevant_chunks)

        print("Yanıt aranıyor...")
        answer = ask_gemini(query, context)
        print(f"\nYanıt: {answer}")

if __name__ == "__main__":
    main()
