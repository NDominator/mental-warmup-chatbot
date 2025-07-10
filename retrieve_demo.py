# retrieve_demo.py
from dotenv import load_dotenv; load_dotenv()           # 1️⃣ read .env
from openai import OpenAI                               # 2️⃣ embed the question
from pinecone import Pinecone                           # 3️⃣ connect to index
import os, textwrap

# ---- Clients -----------------------------------------------------------
client = OpenAI()                                       # uses OPENAI_API_KEY
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
INDEX_NAME = "business-embeddings"
index = pc.Index(INDEX_NAME)

# ---- Ask a question ----------------------------------------------------
question = "How do your pre-game protocols reduce athlete anxiety?"

# 1. Turn the question into an embedding
q_vec = client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
).data[0].embedding

# 2. Query Pinecone for the top 3 similar chunks
matches = index.query(vector=q_vec, top_k=3, include_metadata=True)

# 3. Pretty-print the results
print("\nTop matches:")
for i, m in enumerate(matches["matches"], start=1):
    snippet = textwrap.shorten(m["metadata"]["text"], width=120)
    score   = m["score"]
    print(f"{i}. (score={score:.3f}) {snippet}")
