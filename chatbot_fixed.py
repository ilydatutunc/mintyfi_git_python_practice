import os #dosya\klasör işlemleri
import re #regular expression işlemleri
import json #json dosyası işlemleri
import time #zaman işlemleri
from typing import List, Tuple #liste ve tuple işlemleri
import torch #pytorch işlemleri, derin ögrenme modellerini calistirmak icin
from transformers import AutoTokenizer, AutoModelForCausalLM #transformers kutuphanesi, model yuklemek icin
from sentence_transformers import SentenceTransformer #metin embedding'lerini hesaplamak icin
import faiss #ektor veritabanini olusturmak icin
import pdfplumber #pdf dosyalarini okumak icin



DATA_PATH = "data/nutuk.pdf"
ARTIFACT_DIR = "artifacts" #nutuk ile ilgili olusturacagimiz dosyalar buraya kaydedilecek
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.jsonl")

EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE" #FAISS ile arama yapacağımız metinleri vektöre çevirmek için kullanacağımız model
GEN_MODEL_NAME = "redrussianarmy/gpt2-turkish-cased" #Soruları cevaplamak için kullandığımız text generation modeli

CHUNK_SIZE_WORDS = 500 #her bir chunk'ın kac kelime icermesi gerektigi
CHUNK_OVERLAP_WORDS = 200 #chunk'lar arası kac kelime eslesmesi gerektigi
TOP_K = 4 #en iyi kac cevap alinacagi
MAX_NEW_TOKENS = 120 #cevapta kac kelime gosterilecegi
TEMPERATURE = 0.6
TOP_P = 0.3

os.makedirs(ARTIFACT_DIR, exist_ok=True) #bu klasoru olustur, zaten varsa hata verme

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #device secimi, gpu varsa gpu kullan, yoksa cpu kullan
print(f"Cihaz: {DEVICE}")


def read_pdf_and_clean(pdf_path: str) -> str: #pdf dosyasını temiz bir metin seklinde dondurecek fonksiyon
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text().split("\n")
            merged_lines = []
            buffer = ""
            for line in lines:
                line = line.strip()
                if not line:
                    # boş satır → yeni paragraf
                    if buffer:
                        merged_lines.append(buffer.strip())
                        buffer = ""
                    continue
                if line.endswith("-"):
                    # kelime bölünmüş, tireyi silip boşluk ekleme
                    buffer += line[:-1]
                else:
                    # normal satır → boşluk ekleyerek birlestir
                    buffer += line + " "
            if buffer:
                merged_lines.append(buffer.strip())
                buffer = ""
            text += "\n".join(merged_lines) + "\n"

    # fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> List[str]: #metni "chunk_size" kelime uzunlugunda parcalara ayıracak, overlap kadar kelime eslesmesi
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_size, n)
        chunk_words = words[i:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


#eğer index ve chunks dosyaları varsa yükler. yoksa PDF’yi temizler → parçalar → embedding hesaplar → FAISS’e ekler → diske kaydeder.
def build_or_load_index(pdf_path: str, 
                        embed_model_name: str,
                        index_path: str,
                        chunks_path: str,
                        chunk_size: int = 500,
                        overlap: int = 200,
                        force_rebuild: bool = False) -> Tuple[faiss.IndexFlatIP, List[str], SentenceTransformer]:
    
    embedder = SentenceTransformer(embed_model_name, device=DEVICE)
    embedder.max_seq_length = 512

    if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
        print("[FAISS] Mevcut indeks ve chunk'lar yükleniyor…")
        faiss_index = faiss.read_index(index_path)
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line)["text"])
        return faiss_index, chunks, embedder

    print("[PDF] Nutuk PDF okunuyor ve temizleniyor…")
    clean_text = read_pdf_and_clean(pdf_path)

    print("[DATA] Parçalar oluşturuluyor…")
    chunks = chunk_text_by_words(clean_text, chunk_size, overlap)
    print(f"[DATA] Toplam {len(chunks)} parça üretildi.")

    print("[EMB] Embedding hesaplanıyor…")
    embeddings = embedder.encode(chunks, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"text": ch}, ensure_ascii=False) + "\n")
    print("[FAISS] İndeks ve chunk'lar diske kaydedildi.")
    return index, chunks, embedder


def retrieve(query: str, index: faiss.IndexFlatIP, embedder: SentenceTransformer, chunks: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score), chunks[int(idx)]))
    return results


def load_llm(model_name: str = GEN_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # pad token yoksa eos token kullan
    
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to(DEVICE)
    return tokenizer, model



def build_context_block(retrieved: List[Tuple[int, float, str]]) -> str:
    return "\n\n".join(chunk for _, _, chunk in retrieved)

def generate_answer_fixed(question, tokenizer, model, context_text, temperature=0.2, top_p=0.9, max_new_tokens=120):
    max_context_tokens = 512
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False, max_length=max_context_tokens, truncation=True)
    context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
    
    prompt = f"<|im_start|>user\nAşağıdaki metni kullanarak soruya kısa ve net cevap ver:\n\nMetin: {context_text}\n\nSoru: {question}\n\nCevap:<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=4,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant\n" in raw_answer:
        answer = raw_answer.split("<|im_start|>assistant\n")[-1].strip()
    else:
        answer = raw_answer.strip()
    
    for sep in [". ", "? ", "! "]:
        idx = answer.find(sep)
        if idx != -1:
            answer = answer[:idx+1].strip()
            break
    if not answer or answer.isspace():
        answer = "Üzgünüm, bu soruya cevap veremiyorum. Lütfen soruyu farklı şekilde sorun."
    return answer

# ---------------- RAG pipeline ----------------
def answer_question_rag(question: str, top_k: int = TOP_K, temperature: float = TEMPERATURE, top_p: float = TOP_P, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    start = time.time()
    index, chunks, embedder = build_or_load_index(
        DATA_PATH,
        EMBEDDING_MODEL_NAME,
        INDEX_PATH,
        CHUNKS_PATH,
        CHUNK_SIZE_WORDS,
        CHUNK_OVERLAP_WORDS,
        force_rebuild=False  
    )
    retrieved = retrieve(question, index, embedder, chunks, top_k=top_k)
    context_block = build_context_block(retrieved)
    tokenizer, model = load_llm(GEN_MODEL_NAME)
    answer = generate_answer_fixed(question, tokenizer, model, context_block, temperature, top_p, max_new_tokens)
    elapsed = time.time() - start
    return {
        "question": question,
        "answer": answer,
        "top_k": top_k,
        "retrieved_count": len(retrieved),
        "elapsed_sec": round(elapsed, 2)
    }


def main():
    print("=== Nutuk RAG Chatbot - PDF Temizleme Versiyonu ===")
    test_question = "Nutuk kitabı ne anlatıyor?"
    print(f"Test sorusu: {test_question}")
    print("-" * 50)
    try:
        demo = answer_question_rag(
            test_question,
            top_k=4,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=200,
        )
        print("SORU:", demo["question"])
        print("CEVAP:", demo["answer"])
        print(f"Toplam süre: {demo['elapsed_sec']} saniye")
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Lütfen 'data/nutuk.pdf' dosyasının mevcut olduğundan emin olun.")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()
