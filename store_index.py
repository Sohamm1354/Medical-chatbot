'''from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("C:\\Users\\LENOVO\\Desktop\\End-to-end-Medical-Chatbot-using-Llama2\\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ✅ STEP 1: Manually set credentials and index
PINECONE_API_KEY = "pcsk_6UUUJ8_JDZjgm2w4d5PEhKzLM27ncsa6zkKoJGUJBUoi8NzfrcAtVRnAQCo6DjkQb6VDBm"  # replace with your real key
PINECONE_ENV = "us-east-1"  # e.g., "gcp-starter" or "aws-us-west-2"
index_name = "genai"  # e.g., "medical-chatbot-index"

# ✅ STEP 2: Initialize Pinecone client and create index if needed
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [i.name for i in pc.list_indexes().index_list["indexes"]]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # depends on your embedding model
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"✅ Created new index: {index_name}")
else:
    print(f"✅ Using existing index: {index_name}")

# ✅ STEP 3: Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ STEP 4: Upload text chunks
vectorstore = PineconeVectorStore.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name,
    namespace=""  # optional
)

print("✅ Embeddings uploaded successfully!")
'''
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ✅ Directly set your Pinecone credentials here
PINECONE_API_KEY = "pcsk_6UUUJ8_JDZjgm2w4d5PEhKzLM27ncsa6zkKoJGUJBUoi8NzfrcAtVRnAQCo6DjkQb6VDBm"
PINECONE_ENV = "us-east-1"
index_name = "genai"

# ✅ Load and preprocess PDF
extracted_data = load_pdf("C:\\Users\\LENOVO\\Desktop\\End-to-end-Medical-Chatbot-using-Llama2\\data")
text_chunks = text_split(extracted_data)

# ✅ Download HF embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Pinecone client
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

# ✅ Upload documents to Pinecone
vectorstore = PineconeVectorStore.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name,
    namespace=""
)

print("✅ Embeddings uploaded successfully!")

