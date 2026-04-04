Here’s a clean, professional **README.md** you can directly use for your GitHub repo 👇

---

# 📄 RAG Document Q&A Chatbot (LangChain + Chainlit)

An interactive **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (PDF/TXT) and ask questions based on their content.

Built using **LangChain, Chainlit, and Groq LLM**, this project demonstrates a real-time AI-powered document assistant with streaming responses.

---

## 🚀 Features

* 📂 Upload PDF or Text files
* 🔍 Semantic search using embeddings
* 🤖 AI-powered question answering
* ⚡ Real-time streaming responses
* 🧠 Context-aware answers (RAG pipeline)
* 🛠 Tool-based retrieval using LangChain agents

---

## 🏗️ Tech Stack

* **Frontend/UI**: Chainlit
* **LLM**: Groq (`openai/gpt-oss-120b`)
* **Embeddings**: Google Generative AI (`gemini-embedding-2-preview`)
* **Framework**: LangChain + LangGraph
* **Vector Store**: InMemoryVectorStore

---

## 🧠 How It Works (RAG Pipeline)

1. User uploads a document (PDF/TXT)
2. Document is loaded and split into chunks
3. Chunks are converted into embeddings
4. Stored in an in-memory vector database
5. User asks a question
6. Relevant chunks are retrieved via similarity search
7. LLM generates answer using retrieved context only

---

## 📸 Demo

![Demo](./public/demo.gif)

---

## 📦 Installation

```bash
git clone https://github.com/ShubhamPawar1500/pdf-qa.git
cd pdf-qa

pip install -r requirements.txt
```

---

## ⚙️ Setup

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the App

```bash
chainlit run app.py
```

---

## 📁 Project Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔧 Core Components

### Document Processing

* `PyPDFLoader` / `TextLoader`
* `RecursiveCharacterTextSplitter`

### Embeddings & Storage

* Google Generative AI Embeddings
* In-memory vector store

### Agent System

* Tool-based retrieval (`search_document`)
* LangChain agent with middleware

### UI Layer

* Chainlit for chat interface
* File upload support
* Streaming responses

---

## ⚠️ Limitations

* Uses in-memory vector store (data lost on restart)
* Retrieves only top match (can be improved with top-k)
* No persistent storage
* No reranking

---

## 🚀 Future Improvements

* ✅ Add persistent vector DB (FAISS / Pinecone)
* ✅ Implement top-k retrieval
* ✅ Add reranking (Cohere / cross-encoder)
* ✅ Multi-document support
* ✅ Source citations in responses
* ✅ Authentication & user sessions

---

## 🧑‍💻 Author

**Shubham Pawar**
Full Stack Developer | AI Enthusiast

---

## ⭐ Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---