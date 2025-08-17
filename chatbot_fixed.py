import fitz #pdf'den metin çıkarmak için
import numpy as np #matematiksel işlemler için
from sentence_transformers import SentenceTransformer #metinleri sayısal embedding dönüştürmek için
from sklearn.metrics.pairwise import cosine_similarity #iki vektör arasındaki benzerliği ölçmek için
from transformers import AutoTokenizer, AutoModelForCausalLM #gpt2 modelini yüklemek ve çalıştırmak için
import os #dosya kaydetme / yükleme işlemleri için
import pickle #dosya kaydetme / yükleme işlemleri için
import torch #derin öğrenme modellerini çalıştırmak için

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
def ask_gpt2(query, context, tokenizer, model, max_new_tokens=150):
    # model girişine soru ve bağlam verilir
    inputs = tokenizer(f"Context: {context}\nSoru: {query}\nCevap:", 
                       return_tensors="pt", truncation=True, max_length=1024)
    
    # model cevap üretir (sampling ile biraz rastgelelik eklenmiştir)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # yanıt metni decode edilir
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # sadece "cevap:" kısmından sonrası alınır
    answer = answer.split("Cevap:")[-1].strip()
    return answer if answer else "Cevap bulunamadı."

# context çok uzun olduğunda token sayısını azaltır
# modelin max_token limitini aşmamak için eski kısımları atar
def truncate_context(query, context, tokenizer, max_tokens=800):
    context_tokens = tokenizer.encode(context)
    query_tokens = tokenizer.encode(query)
    if len(context_tokens) + len(query_tokens) > max_tokens:
        context_tokens = context_tokens[-(max_tokens - len(query_tokens)):]
    truncated_context = tokenizer.decode(context_tokens)
    return truncated_context

# ana çalıştırma fonksiyonu
# tüm pipeline burada çalışır:
# 1) embedding modeli yüklenir
# 2) pdf işlenir (cache varsa yüklenir)
# 3) gpt2 instruct modeli yüklenir
# 4) kullanıcıdan soru alınır, ilgili chunk'lar bulunur, cevap üretilir
def main():
    pdf_path = "data/nutuk.pdf"

    # embedding modeli yükleniyor
    print("Embedding modeli yükleniyor (CPU)...")
    embed_model = SentenceTransformer("trmteb/turkish-embedding-model", device="cpu")

    # cache yükleniyor veya yeniden oluşturuluyor
    chunks, embeddings = load_or_create_cache(pdf_path, embed_model)
    
    # gpt2 instruct modeli yükleniyor
    print("GPT2 Instruction modeli ve tokenizer yükleniyor (CPU)...")
    model_name = "ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print("Model başarıyla yüklendi!")
    
    # kullanıcıdan sürekli olarak soru alınıp cevap üretilir
    while True:
        query = input("\nSorunuzu yazın (çıkmak için 'quit'): ")
        if query.lower() == 'quit':
            break

        # soruya en uygun chunkslari bulunur
        relevant_chunks = find_relevant_chunks(query, chunks, embeddings, embed_model)
        context = "\n\n".join(relevant_chunks)
        context = truncate_context(query, context, tokenizer)

        print("Yanıt aranıyor...")
        try:
            # modelden yanıt alınır
            answer = ask_gpt2(query, context, tokenizer, model)
            print(f"\nYanıt: {answer}")
        except Exception as e:
            print(f"Hata: {e}")


if __name__ == "__main__":
    main()
