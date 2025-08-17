import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pickle

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Nutuk Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .info-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_or_create_cache(pdf_path, _embed_model, cache_dir="cache"):
    """Cache'den yükle veya oluştur"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    chunks_cache = os.path.join(cache_dir, "nutuk_chunks.pkl")
    embeddings_cache = os.path.join(cache_dir, "nutuk_embeddings.pkl")
    
    if os.path.exists(chunks_cache) and os.path.exists(embeddings_cache):
        with open(chunks_cache, "rb") as f:
            chunks = pickle.load(f)
        with open(embeddings_cache, "rb") as f:
            embeddings = pickle.load(f)
        return chunks, embeddings
    
    # Cache yoksa oluştur
    with st.spinner("🔄 Cache oluşturuluyor..."):
        # PDF'den metin çıkar
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Metni chunk'lara böl
        chunks = []
        chunk_size = 500
        overlap = 100
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        
        # Embedding'ler oluştur
        embeddings = _embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        
        # Cache'e kaydet
        with open(chunks_cache, "wb") as f:
            pickle.dump(chunks, f)
        with open(embeddings_cache, "wb") as f:
            pickle.dump(embeddings, f)
    
    return chunks, embeddings

@st.cache_data
def load_models():
    """Modelleri yükle"""
    with st.spinner("🤖 Modeller yükleniyor..."):
        # Embedding modeli
        embed_model = SentenceTransformer("trmteb/turkish-embedding-model", device="cpu")
        
        # GPT2 Instruction modeli
        model_name = "ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        return embed_model, tokenizer, model

def find_relevant_chunks(query, chunks, embeddings, embed_model, top_k=3):
    """Sorguya en uygun chunk'ları bul"""
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def truncate_context(query, context, tokenizer, max_tokens=800):
    """Context'i token limitine göre kır"""
    context_tokens = tokenizer.encode(context)
    query_tokens = tokenizer.encode(query)
    if len(context_tokens) + len(query_tokens) > max_tokens:
        context_tokens = context_tokens[-(max_tokens - len(query_tokens)):]
    truncated_context = tokenizer.decode(context_tokens)
    return truncated_context

def ask_gpt2(query, context, tokenizer, model, max_new_tokens=150):
    """GPT2 Instruction ile yanıt üret"""
    try:
        inputs = tokenizer(f"Context: {context}\nSoru: {query}\nCevap:", 
                          return_tensors="pt", truncation=True, max_length=1024)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Cevap:")[-1].strip()
        return answer if answer else "Cevap bulunamadı."
        
    except Exception as e:
        return f"Nutuk metninden bulunan bilgi: {context[:200]}..."

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">📚 Nutuk Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">ℹ️ Hakkında</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Bu chatbot, Mustafa Kemal Atatürk'ün **Nutuk** eserini kullanarak 
        sorularınızı yanıtlar.
        
        **Nasıl çalışır:**
        1. PDF'den metin çıkarılır ve chunk'lara bölünür
        2. Her chunk için embedding oluşturulur
        3. Sorgunuz en uygun chunk'larla eşleştirilir
        4. Türkçe GPT2 Instruction modeli yanıt üretir
        """)
        
        st.markdown("---")
        
        # Örnek sorular
        st.markdown("**💡 Örnek Sorular:**")
        example_questions = [
            "Atatürk Samsun'a ne zaman çıkmıştır?",
            "Kurtuluş Savaşı nasıl başladı?",
            "TBMM ne zaman kuruldu?",
            "Atatürk'ün liderlik özellikleri nelerdir?",
            "İzmir'in kurtuluşu ne zaman oldu?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_input = question
                st.rerun()
    
    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Sohbet")
        
        # Chat geçmişi
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mesajları göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Kullanıcı girişi
        if prompt := st.chat_input("Nutuk hakkında bir soru sorun..."):
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Bot yanıtını hazırla
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    try:
                        # Modelleri yükle
                        embed_model, tokenizer, model = load_models()
                        
                        # Cache'den yükle
                        chunks, embeddings = load_or_create_cache("data/nutuk.pdf", embed_model)
                        
                        # İlgili chunk'ları bul
                        relevant_chunks = find_relevant_chunks(prompt, chunks, embeddings, embed_model)
                        context = "\n\n".join(relevant_chunks)
                        
                        # Context'i kır
                        context = truncate_context(prompt, context, tokenizer)
                        
                        # GPT2 ile yanıt üret
                        response = ask_gpt2(prompt, context, tokenizer, model)
                        
                        # Yanıtı göster
                        st.markdown(response)
                        
                        # Bot mesajını ekle
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Context'i session'a kaydet
                        st.session_state.last_context = context
                        
                    except Exception as e:
                        st.error(f"Bir hata oluştu: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Hata: {str(e)}"})
    
    with col2:
        st.markdown("### 📊 İstatistikler")
        
        try:
            # Modelleri yükle
            embed_model, tokenizer, model = load_models()
            
            # Cache'den yükle
            chunks, embeddings = load_or_create_cache("data/nutuk.pdf", embed_model)
            
            st.metric("Toplam Chunk", len(chunks))
            st.metric("Chunk Boyutu", "500 karakter")
            st.metric("Embedding Boyutu", f"{embeddings.shape[1]} boyut")
            
            # Son kullanılan context'i göster
            if "last_context" in st.session_state:
                st.markdown("**🔍 Son Kullanılan Bağlam:**")
                st.text_area("", value=st.session_state.last_context, height=200, disabled=True)
                
        except Exception as e:
            st.error("Modeller yüklenemedi")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Bu chatbot, Mustafa Kemal Atatürk'ün Nutuk eserini kullanarak oluşturulmuştur.</p>
        <p>Teknoloji: Streamlit + Türkçe GPT2 Instruction + Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
