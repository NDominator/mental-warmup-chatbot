# embedding_demo.py
# 1) Load environment keys
from dotenv import load_dotenv           # reads the .env file
load_dotenv()

# 2) Import the OpenAI client
from openai import OpenAI
client = OpenAI()                        # picks up OPENAI_API_KEY automatically

# 3) Text we’ll embed
text = "Pregame protocols build confident competitors."

# 4) Request an embedding vector from OpenAI
response = client.embeddings.create(
    input=text,                          # the text to turn into numbers
    model="text-embedding-3-small"       # cheap, good-enough embedding model
)

# 5) Print the first 10 numbers (there are ~1536 total)
vector = response.data[0].embedding
print("First 10 numbers of the embedding:", vector[:10])
