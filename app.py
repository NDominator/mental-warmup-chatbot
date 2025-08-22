# app.py  – Step 1
import streamlit as st                # Streamlit turns Python into a web app

st.title("Mental Warmup Chatbot")      # Big heading
st.write("Let's build a mental warmup!")  # intro text

# --- Set up session state for chat history ---
if "history" not in st.session_state:
    st.session_state["history"] = []   # list of {"q": ..., "a": ...}
# --- Clear-chat button ---
if st.button("🗑️  Clear chat history"):
    st.session_state["history"] = []     # empty the list
    if hasattr(st, "rerun"):         # Streamlit ≥ 1.28
        st.rerun()
    else:                            # older versions
        st.experimental_rerun()


# --- RAG setup (runs once when the app starts) ---------------------------
from dotenv import load_dotenv; load_dotenv()          # read .env secrets
import os
from pinecone import Pinecone                          # connect to Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore     # v3-compatible adapter
from langchain.chains import RetrievalQA

# 1️⃣  Connect to your Pinecone index
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
INDEX_NAME = "mental-warmup-bot"

# 2️⃣  Set up embeddings + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text"            # where we stored the chunk text
)

# 3️⃣  Build the Retrieval-QA chain
chat_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# --- Chat interaction with history --------------------------------------
# Give the text box a key so we can clear it later
user_question = st.text_input(
    "Your question:",
    placeholder="Type here and press Enter…",
    key="user_input"
)

if user_question:
    with st.spinner("Thinking…"):
    # ⬇️ use .invoke(), which always returns source_documents
        result  = qa_chain.invoke(
        {"query": user_question, "return_source_documents": True}
    )
    answer  = result["result"]
    sources = result.get("source_documents", [])   # safe fallback


        # 1️⃣ save this turn
    st.session_state.history.append({
            "q": user_question,
            "a": answer,
            "src": sources if sources else []
        })


        # 2️⃣ (optional) you could clear the box with a callback,
        #    but we'll skip it for now to avoid the session_state error.


# 3️⃣ show the entire chat log, newest first
for turn in reversed(st.session_state.history):
    st.markdown(f"**You:** {turn['q']}")
    st.markdown(f"**Coach:** {turn['a']}")
    if turn.get("src"):
        if st.checkbox("Show retrieved chunks", key=f"chk_{id(turn)}"):
            for i, doc in enumerate(turn["src"], start=1):
                snippet = doc.page_content[:300].replace("\n", " ")
                st.markdown(f"*Chunk {i}:* {snippet}…")
    st.markdown("---")

