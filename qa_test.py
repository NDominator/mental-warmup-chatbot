from dotenv import load_dotenv; load_dotenv()
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs = PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    index_name="business-embeddings",
    embedding=embeddings,
    text_key="text"
)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vs.as_retriever(search_kwargs={"k": 3})
)

q = "What is a PGP?"
print("Sending:", q)
a = qa.invoke({"query": q})["result"]      # new API; .run is deprecated
print("Answer:", a)
