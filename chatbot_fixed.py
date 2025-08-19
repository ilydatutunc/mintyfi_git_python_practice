import fitz #pdf'den metin çıkarmak için
import numpy as np #matematiksel işlemler için
from sentence_transformers import SentenceTransformer #metinleri sayısal embedding dönüştürmek için
from sklearn.metrics.pairwise import cosine_similarity #iki vektör arasındaki benzerliği ölçmek için
from transformers import AutoTokenizer, AutoModelForCausalLM #gpt2 modelini yüklemek ve çalıştırmak için
import os #dosya kaydetme / yükleme işlemleri için
import pickle #dosya kaydetme / yükleme işlemleri için
import torch #derin öğrenme modellerini çalıştırmak için
import re #regex işlemleri için (tarih yakalama)

# pdf dosyasından tüm metni çıkarır
# her sayfanın içindeki metni alıp tek bir string haline getirir
def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# uzun metni küçük chunkslara böler
# chunk_size: bir parçanın maksimum uzunluğu
# overlap: parçalar arasında tekrarlanan kısım (bağlamın kopmaması için)
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks

# cache kontrolü yapar
# eğer daha önce pdf işlenmişse, chunks ve embedding'ler diskte saklanır
# böylece her çalıştırmada tekrar hesaplama yapılmaz
def load_or_create_cache(pdf_path, embed_model, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    chunks_cache = os.path.join(cache_dir, "nutuk_chunks.pkl")
    embeddings_cache = os.path.join(cache_dir, "nutuk_embeddings.pkl")
    
    # eğer cache dosyaları varsa doğrudan yükle
    if os.path.exists(chunks_cache) and os.path.exists(embeddings_cache):
        print("Cache'den yükleniyor...")
        with open(chunks_cache, "rb") as f:
            chunks = pickle.load(f)
        with open(embeddings_cache, "rb") as f:
            embeddings = pickle.load(f)
        print(f"{len(chunks)} chunk ve embedding'ler cache'den yüklendi!")
        return chunks, embeddings
    
    # cache yoksa pdf'i işleyip yeni oluştur
    print("Cache oluşturuluyor...")
    text = pdf_to_text(pdf_path)
    chunks = split_text(text)
    print(f"Toplam {len(chunks)} chunk oluşturuldu.")
    
    # her chunk için embedding hesaplanır (vektör temsili)
    print("Embedding modeli ile embedding'ler oluşturuluyor...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # oluşturulan veriler diske kaydedilir
    with open(chunks_cache, "wb") as f:
        pickle.dump(chunks, f)
    with open(embeddings_cache, "wb") as f:
        pickle.dump(embeddings, f)
    
    print("Cache oluşturuldu ve kaydedildi!")
    return chunks, embeddings

# verilen bir sorguya (query) en uygun metin parçalarını bulur
# önce sorgu embedding'e çevrilir
# sonra cosine similarity ile chunk'larla karşılaştırılır
# en çok benzeyen top_k adet chunk seçilir
def find_relevant_chunks(query, chunks, embeddings, embed_model, top_k=3):
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# gpt2 instruct modeli ile yanıt üretir
# context: pdf'den seçilen parçalar
# query: kullanıcının sorusu
# model generate fonksiyonu ile yeni metin üretir
def extract_date(answer):
    # ay isimleriyle tarih yakalama
    patterns = [
        r"\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}\b",
        r"\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'de\b",
        r"\b\d{4}\s+yılında\b",
        r"\b\d{4}\s+yılı\b"
    ]
    
    # her pattern'i dene
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(0)
    
    return "Cevap bulunamadı"

def ask_gpt2(query, context, tokenizer, model, max_new_tokens=50):
    try:
        # Tarih sorusu mu kontrol et
        is_date_question = any(word in query.lower() for word in [
            'ne zaman', 'tarih', 'yıl', 'yılında', 'kaçıncı', 'hangi tarih'
        ])
        
        # Context'i güvenli hale getir
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        if is_date_question:
            # tarih soruları için daha sıkı prompt
            prompt_template = f"""
Sen bir metin analiz aracısın. Görevin sadece metinde geçen SORU’ya ait TARİH bilgisini çıkarmaktır. 
Hiçbir yorum, tahmin veya ek bilgi verme. 
Eğer metinde tarih yoksa sadece "Cevap bulunamadı" yaz. 
YANIT: Sadece tarih veya yıl formatında yaz (ör: 1 Kasım 1922 veya 1923). Başka hiçbir kelime ekleme.

METİN:
{context}

SORU:
{query}

YANIT:"""
            do_sample = True
            temperature = 0.4
            top_p = 0.9
            top_k = 15
        else:
            # genel sorular için prompt
            prompt_template = f"""
Sen bir metin analiz aracısın. Aşağıdaki Nutuk metni ile sınırlısın. 
Görevin sadece SORU’daki sorunun cevabını metinden çıkarmaktır. . 
Eğer metinde cevap yoksa sadece "Cevap bulunamadı" yaz. 
Yanıtını tek cümle, kısa ve net olarak ver, gereksiz kelime kullanma.
METİN:
{context}

SORU:
{query}

YANIT:"""
            do_sample = True
            temperature = 0.6
            top_p = 0.95
            top_k = 30
        
        # Tokenize et
        inputs = tokenizer(prompt_template, return_tensors="pt", truncation=True, max_length=1024)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # yanıtı decode et
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # prompt'u yanıttan çıkar
        if "YANIT:" in answer:
            answer = answer.split("YANIT:")[-1].strip()
        
        # Tarih sorularında regex ile filtre uygula
        if is_date_question:
            answer = extract_date(answer)
        
        # Eğer hala boşsa fallback
        if not answer:
            answer = "Cevap bulunamadı."
        
        return answer
    
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


# context çok uzun olduğunda token sayısını azaltır
# modelin max_token limitini aşmamak için eski kısımları atar
def truncate_context(query, context, tokenizer, max_tokens=1000):
    context_tokens = tokenizer.encode(context)
    query_tokens = tokenizer.encode(query)
    if len(context_tokens) + len(query_tokens) > max_tokens:
        context_tokens = context_tokens[-(max_tokens - len(query_tokens)):]
    truncated_context = tokenizer.decode(context_tokens)
    return truncated_context

# Embedding ve GPT2 Instruction modellerini yükler
def load_models():
    print("Embedding modeli yükleniyor (CPU)...")
    embed_model = SentenceTransformer("trmteb/turkish-embedding-model", device="cpu")

    print("GPT2 Instruction modeli ve tokenizer yükleniyor (CPU)...")
    model_name = "ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return embed_model, tokenizer, model

# ana çalıştırma fonksiyonu
# tüm pipeline burada çalışır:
# 1) embedding modeli yüklenir
# 2) pdf işlenir (cache varsa yüklenir)
# 3) gpt2 instruct modeli yüklenir
# 4) kullanıcıdan soru alınır, ilgili chunk'lar bulunur, cevap üretilir
def main():
    pdf_path = "data/nutuk.pdf"

    embed_model, tokenizer, model = load_models()
    chunks, embeddings = load_or_create_cache(pdf_path, embed_model)

    while True:
        query = input("\nSorunuzu yazın (çıkmak için 'quit'): ")
        if query.lower() == 'quit':
            break

        relevant_chunks = find_relevant_chunks(query, chunks, embeddings, embed_model)
        context = "\n\n".join(relevant_chunks)
        context = truncate_context(query, context, tokenizer)

        print("Yanıt aranıyor...")
        try:
            answer = ask_gpt2(query, context, tokenizer, model)
            print(f"\nYanıt: {answer}")
        except Exception as e:
            print(f"Hata: {e}")


if __name__ == "__main__":
    main()