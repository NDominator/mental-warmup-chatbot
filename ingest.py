# ingest.py – Part 1
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()                 # ← OK

# (no pinecone import or init call here anymore)

# ingest.py – Part 2  🗄️  Load every file in docs/
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader   # PDF & text support

all_docs = []                                     # will hold every chunk we load

for path in Path("docs").iterdir():               # loop over each file in docs/
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))           # 1️⃣ read a PDF, page by page
        all_docs.extend(loader.load())            #    add pages to all_docs
    elif ext in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")  # 2️⃣ read plaintext
        all_docs.extend(loader.load())
    # you can add more `elif` branches later for other file types

print(f"Loaded {len(all_docs)} document chunks.")

# ingest.py – Part 3  ✂️  Split into short, overlapping chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # characters per chunk (~250–300 words)
    chunk_overlap=200  # repeats 200 chars so context isn’t lost
)

chunks = splitter.split_documents(all_docs)       # returns a new list
print(f"Split into {len(chunks)} smaller chunks.")

# ingest.py – Part 4  🗄️  Ensure Pinecone index exists (v3 syntax, with spec)
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

INDEX_NAME = "business-embeddings"
EMBED_DIM  = 1536

if INDEX_NAME not in pc.list_indexes():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created Pinecone index '{INDEX_NAME}'.")
else:
    print(f"Pinecone index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)     # handle for upserts/queries

# ingest.py – Part 5  🧠  Create embeddings & upsert to Pinecone
from uuid import uuid4                               # makes unique IDs

batch = []                                           # holds up to 100 items
for i, doc in enumerate(chunks, start=1):
    # 1️⃣ Turn the chunk text into a vector
    vec = client.embeddings.create(
        input=doc.page_content,
        model="text-embedding-3-small"
    ).data[0].embedding

    # 2️⃣ Build the Pinecone record
    record = {
        "id": str(uuid4()),                          # unique ID
        "values": vec,                               # the number list
        "metadata": {
            "text": doc.page_content,  # store full text!
            "source": doc.metadata.get("source", "")
        }
    }
    batch.append(record)

    # 3️⃣ Every 100 records, send them to Pinecone
    if len(batch) == 100:
        index.upsert(batch)
        batch = []
        print(f"Upserted {i} / {len(chunks)} chunks...")

# 4️⃣ Push any leftovers
if batch:
    index.upsert(batch)

print("✅ All embeddings upserted to Pinecone!")
