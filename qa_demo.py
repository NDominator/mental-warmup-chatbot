# qa_demo.py
# 1️⃣ Load secrets
from dotenv import load_dotenv; load_dotenv()

# 2️⃣ LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI      # embeddings + chat
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# 3️⃣ Pinecone client
import os
from pinecone import Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

INDEX_NAME = "business-embeddings"

# 4️⃣ Plug Pinecone into LangChain’s VectorStore wrapper
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),   # let adapter connect
    index_name=INDEX_NAME,                            # the index we created
    embedding=embeddings,
    text_key="text"                                   # metadata field for the chunk
)


# 5️⃣ Build the Retrieval-QA chain
chat_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    chain_type="stuff",          # simplest: dump retrieved chunks into prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# 6️⃣ Ask a real question
question = "How do your pre-game protocols help athletes feel confident?"
answer = qa_chain.run(question)

print("Q:", question)
print("A:", answer)
