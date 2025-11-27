import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

# ------------------- 설정 -------------------

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)      # 또는 qwen2.5:7b, gemma2:9b 등
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

DB_PATH = "./chroma_db"

# ------------------- 벡터 DB 초기화 (핵심 수정 부분) -------------------
if os.path.exists(DB_PATH) and any(Path(DB_PATH).iterdir()):
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    st.sidebar.success("기존 벡터 DB를 로드했습니다.")
else:
    # 새로 만들 때
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 샘플 문서가 있으면 자동 인덱싱
    if os.path.exists("./documents") and os.listdir("./documents"):
        loader = DirectoryLoader(
            "./documents",
            glob="**/*",
            loader_cls=lambda path: PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding="utf-8")
        )
        docs = loader.load()
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # 여기서 documents= 로 수정! ← 이게 핵심
            vectorstore = Chroma.from_documents(
                documents=splits,                # ← docs → documents
                embedding=embeddings,            # ← embedding_function이 아니라 embedding
                persist_directory=DB_PATH
            )
            st.sidebar.success("샘플 문서를 인덱싱했습니다.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------- 문서 업로드 & 추가 (이 부분도 안전하게 수정) -------------------
st.sidebar.header("문서 추가 (txt, pdf 지원)")
uploaded_files = st.sidebar.file_uploader(
    "문서를 업로드하면 바로 벡터 DB에 추가됩니다",
    accept_multiple_files=True,
    type=["txt", "pdf"]
)

if uploaded_files:
    docs = []
    temp_dir = "./temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(file_path) if uploaded_file.name.lower().endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load())
    
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore.add_documents(documents=splits)   # ← 여기서도 documents= 로 안전하게
        st.sidebar.success(f"{len(uploaded_files)}개 문서가 추가되었습니다!")
    
    shutil.rmtree(temp_dir, ignore_errors=True)

# ------------------- 챗봇 인터페이스 (변경 없음) -------------------
st.title("RAG 챗봇")
st.caption("문서를 업로드하고, 내용에 대해 질문하세요.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(answer)

            if sources:
                st.markdown("**참고한 문서 조각**")
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**출처 {i}:** {doc.page_content[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
