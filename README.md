# End-to-end-Medical-Chatbot-using-Llama2

🩺 End-to-End Medical Chatbot using LLaMA2, LangChain, HuggingFace & Pinecone
This project is an AI-powered medical chatbot built using LangChain, LLaMA2 (via CTransformers), HuggingFace sentence embeddings, and Pinecone vector store. It enables users to ask medical-related queries, and provides context-aware, relevant answers by retrieving information from PDF-based medical documents.

🔍 Features
✅ Extracts and processes medical data from PDF documents

✅ Splits documents into meaningful chunks for better retrieval

✅ Embeds text using HuggingFace Transformers

✅ Stores and indexes embeddings using Pinecone

✅ Retrieves top relevant document chunks for user queries

✅ Uses LLaMA2 language model for generating informative answers

✅ Integrates a web-based chat interface using Flask

| Technology                 | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| **LangChain**              | Prompt management, document loading, QA chain |
| **LLaMA2 (CTransformers)** | Local LLM for generating answers              |
| **HuggingFace Embeddings** | Semantic text embeddings                      |
| **Pinecone**               | Vector storage and similarity search          |
| **Flask**                  | Web server for user interaction               |
| **HTML/CSS**               | Frontend for the chatbot interface            |



