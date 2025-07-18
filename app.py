import streamlit as st
import os
import tempfile
import requests
from utils import *
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# print("QDRANT_URL:", QDRANT_URL)
# print("QDRANT_API:", QDRANT_API)
# print("OPENROUTER_API_KEY:", OPENROUTER_API_KEY)

# Initialize client and embeddings
try:

    # Streamlit UI
    st.set_page_config(layout="wide")
    st.title("Simple RAG Q&A")

    # Sidebar: upload or URL
    st.sidebar.header("Upload PDF or URL (max 10MB)")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    pdf_url = st.sidebar.text_input("Or enter PDF URL")

    st.sidebar.info("Select number of output tokens (default 512)")
    output_tokens = st.sidebar.number_input("Output tokens", min_value=512, max_value=1024, value=512, step=64)

    # Handle file upload
    if uploaded_file:
        size = uploaded_file.size
        if size > 10 * 1024 * 1024:
            st.sidebar.error("File exceeds 10MB")
        else:
            with st.spinner("Processing file..."):
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                process_pdf(temp_path, uploaded_file.name)
                st.sidebar.success("File processed and stored")

    # Handle URL
    if pdf_url:
        if not pdf_url.lower().endswith('.pdf'):
            st.sidebar.warning("URL may not point to a PDF")
        try:
            resp = requests.get(pdf_url, stream=True, timeout=10)
            resp.raise_for_status()
            size = int(resp.headers.get('content-length', 0))
            if size > 10 * 1024 * 1024:
                st.sidebar.error("PDF exceeds 10MB")
            else:
                with st.spinner("Downloading and processing..."):
                    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                    with open(temp_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                    process_pdf(temp_path, os.path.basename(pdf_url))
                    st.sidebar.success("URL PDF processed and stored")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    # Main area: question
    question = st.text_input("Ask a question about the uploaded PDFs:")
    if st.button("Ask") and question:
        # Setup QA
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question.
            {context}
            Question: {question}
            """
        )
        llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-chat:free",
            max_completion_tokens=output_tokens,
        )
        docs = vectorstore.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docs,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        with st.spinner("Generating answer..."):
            result = qa.invoke(question)
        st.subheader("Answer:")
        answer = result.get("result", "No answer found.")
        sources = result.get("source_documents", [])
        for doc in sources:
            print(doc.metadata)
        source_texts = [f"(Page {doc.metadata.get('page', 0)})" for doc in sources]
        # st.write(result)
        st.markdown(f"**Answer:** {answer}")
        if sources:
            st.markdown("**Sources:**")
            for i, doc in enumerate(sources):
                st.markdown(f"{i + 1}. {doc.metadata.get('source_file', 'Unknown')} {source_texts[i]}")
        
except KeyboardInterrupt:
    st.sidebar.error("Process interrupted.")
    print("Process interrupted by user.")
    vectorstore.client.close()
    qdrant_client.close()
    st.stop()
