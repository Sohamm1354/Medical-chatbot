from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone

import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from typing import List, Any
from langchain_core.documents import Document
from pydantic import Field
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from src.prompt import *  # 🔧 Removed dotenv
import os

app = Flask(__name__)

# 🔧 Step 1: Directly set API Key and Environment
PINECONE_API_KEY = "pcsk_6UUUJ8_JDZjgm2w4d5PEhKzLM27ncsa6zkKoJGUJBUoi8NzfrcAtVRnAQCo6DjkQb6VDBm"
PINECONE_ENV = "us-east-1"
index_name = "genai"

# ✅ Set env vars for PineconeVectorStore
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENV
# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load PDF files
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

extracted_data = load_pdf("C:\\Users\\LENOVO\\Desktop\\End-to-end-Medical-Chatbot-using-Llama2\\data")

# Split text
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

# 🔧 Step 2: Initialize Pinecone manually
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [i.name for i in pc.list_indexes().index_list["indexes"]]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"✅ Created new index: {index_name}")
else:
    print(f"✅ Using existing index: {index_name}")

# 🔧 Step 3: Upload embeddings
'''
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name,
    namespace=""
)
print("✅ Embeddings uploaded successfully!")
'''
# 🔧 Step 4: Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Prompt setup
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load LLM
llm = CTransformers(
    model="model",
    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.8,
    }
)

# Custom Retriever
class CustomRetriever(BaseRetriever):
    vectorstore: Any = Field()

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=2)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await self.vectorstore.asimilarity_search(query, k=2)

retriever = CustomRetriever(vectorstore=docsearch)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    result = qa.invoke({"query": msg})
    print("Response:", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
