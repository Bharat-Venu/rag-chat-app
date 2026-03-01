## backend.py — FastAPI Backend for RAG PDF Chat
## Run with: uvicorn backend:app --reload

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile, os, shutil
from typing import List

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI()

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state: session_id -> { chain, store }
sessions: dict = {}

# Load embeddings once at startup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    api_key: str = Form(...),
    session_id: str = Form("default_session"),
):
    try:
        documents = []
        for uploaded_file in files:
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(uploaded_file.file, tmp)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
            os.unlink(tmp_path)

        # Split & embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # LLM
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might reference context "
             "in the chat history, formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, just reformulate if needed."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant for question-answering tasks. Use the following pieces of "
             "retrieved context to answer the question. If you don't know the answer, say that "
             "you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Session store
        store = {}

        def get_session_history(sid: str) -> BaseChatMessageHistory:
            if sid not in store:
                store[sid] = ChatMessageHistory()
            return store[sid]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        sessions[session_id] = {"chain": conversational_chain, "store": store}
        return {"success": True, "message": f"Processed {len(documents)} pages"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


class ChatRequest(BaseModel):
    question: str
    api_key: str
    session_id: str = "default_session"


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        return JSONResponse(status_code=400, content={"error": "No documents uploaded for this session."})
    try:
        chain = sessions[req.session_id]["chain"]
        response = chain.invoke(
            {"input": req.question},
            config={"configurable": {"session_id": req.session_id}},
        )
        return {"answer": response["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
def health():
    return {"status": "ok"}
