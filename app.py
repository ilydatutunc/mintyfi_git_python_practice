import streamlit as st
from chatbot_fixed import load_models, load_or_create_cache, find_relevant_chunks, truncate_context, ask_gpt2

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

def main():
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
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Nutuk hakkında bir soru sorun..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    try:
                        embed_model, tokenizer, model = load_models()
                        chunks, embeddings = load_or_create_cache("data/nutuk.pdf", embed_model)
                        relevant_chunks = find_relevant_chunks(prompt, chunks, embeddings, embed_model)
                        context = "\n\n".join(relevant_chunks)
                        context = truncate_context(prompt, context, tokenizer)
                        response = ask_gpt2(prompt, context, tokenizer, model)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.last_context = context
                    except Exception as e:
                        st.error(f"Bir hata oluştu: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Hata: {str(e)}"})

    with col2:
        st.markdown("### 📊 İstatistikler")
        try:
            embed_model, tokenizer, model = load_models()
            chunks, embeddings = load_or_create_cache("data/nutuk.pdf", embed_model)
            st.metric("Toplam Chunk", len(chunks))
            st.metric("Chunk Boyutu", "500 karakter")
            st.metric("Embedding Boyutu", f"{embeddings.shape[1]} boyut")
            if "last_context" in st.session_state:
                st.markdown("**🔍 Son Kullanılan Bağlam:**")
                st.text_area("", value=st.session_state.last_context, height=200, disabled=True)
        except Exception as e:
            st.error("Modeller yüklenemedi")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Bu chatbot, Mustafa Kemal Atatürk'ün Nutuk eserini kullanarak oluşturulmuştur.</p>
        <p>Teknoloji: Streamlit + Türkçe GPT2 Instruction + Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
