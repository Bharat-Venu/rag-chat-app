# docRAG — PDF Chat Web App

A full-stack web app that replaces your Streamlit RAG app with a React frontend + FastAPI backend.

---

## 🗂 Files

```
rag-chat-app/
├── frontend.jsx   ← React component (paste into your React app)
├── backend.py     ← FastAPI server
└── README.md
```

---

## ⚙️ Backend Setup

### 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart langchain langchain-groq \
    langchain-huggingface langchain-community faiss-cpu pypdf \
    sentence-transformers python-dotenv
```

### 2. Run the server
```bash
uvicorn backend:app --reload --port 8000
```

---

## ⚛️ Frontend Setup (React)

### Option A — Vite (recommended)
```bash
npm create vite@latest rag-frontend -- --template react
cd rag-frontend
npm install
# Replace src/App.jsx contents with frontend.jsx
npm run dev
```

### Option B — Paste into Claude.ai Artifacts
Copy the entire `frontend.jsx` and paste into a new artifact. Change `API_BASE` to match your backend URL.

---

## 🌐 Key Differences from Streamlit

| Streamlit | Web App |
|-----------|---------|
| `st.text_input` | `<input>` controlled component |
| `st.file_uploader` | Drag-and-drop + file input |
| `st.session_state` | React `useState` / FastAPI session dict |
| `st.write` | Chat bubbles with history |
| Re-runs on every interaction | Event-driven, no full-page reload |

---

## 🔒 Production Notes
- Set `allow_origins` in CORS to your actual domain
- Store API keys in `.env`, never commit them
- Use a persistent vector DB (Chroma, Pinecone) instead of in-memory FAISS for multi-user setups
- Deploy backend on Railway, Render, or EC2; frontend on Vercel/Netlify
