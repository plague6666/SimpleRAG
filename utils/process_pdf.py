from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import vectorstore

def process_pdf(path: str, source: str):
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = source
        splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)