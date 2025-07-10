# update_index.py – Part 1
# ① Load .env secrets
from dotenv import load_dotenv
load_dotenv()

# ② Imports
import os, hashlib                       # os for env vars, hashlib for md5 IDs
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException  # generic API errors
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ③ Connect / create index if needed
# ---- Robust index setup (works even if it already exists) ---------------
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException  # generic API errors

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

INDEX_NAME = "business-embeddings"
EMBED_DIM  = 1536

try:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"🆕 Created index '{INDEX_NAME}'.")
except PineconeApiException as e:
    if e.status == 409:   # “Resource already exists”
        print(f"🗂️  Index '{INDEX_NAME}' already exists – skipping create.")
    else:
        raise            # bubble up any other error

index = pc.Index(INDEX_NAME)   # handle for read/write

# update_index.py – Part 2  🗄️  Load + split docs
loaders = []
for path in Path("docs").iterdir():
    ext = path.suffix.lower()
    if ext == ".pdf":
        loaders.append(PyPDFLoader(str(path)))
    elif ext in (".txt", ".md"):
        loaders.append(TextLoader(str(path), encoding="utf-8"))

all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())       # 1 doc per page (PDF) or per file (txt)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
chunks = splitter.split_documents(all_docs)

print(f"🔍 Prepared {len(chunks)} chunks for upsert.")

# update_index.py – Part 3  🔼 Embed & upsert
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

batch = []
for i, doc in enumerate(chunks, start=1):
    text   = doc.page_content
    vec    = embeddings.embed_query(text)      # returns the 1536-number vector
    doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()  # deterministic ID

    record = {
        "id": doc_id,
        "values": vec,
        "metadata": {
            "text": text[:200],                # preview for dashboard
            "source": doc.metadata.get("source", "")
        }
    }
    batch.append(record)

    if len(batch) == 100:
        index.upsert(batch)
        print(f"➡️  Upserted {i}/{len(chunks)} chunks…")
        batch = []

if batch:                                      # push any leftovers
    index.upsert(batch)

print("✅ Incremental upsert complete!")
