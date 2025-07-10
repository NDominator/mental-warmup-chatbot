# similarity_demo.py
from dotenv import load_dotenv; load_dotenv()      # 1️⃣ read .env for your keys
from openai import OpenAI                          # 2️⃣ import OpenAI client
import math                                        # 3️⃣ built-in math for cosine
client = OpenAI()                                  # 4️⃣ auth via OPENAI_API_KEY

# 5️⃣ Two sentences to compare
text1 = "Pregame protocols build confident competitors."
text2 = "These routines help athletes feel calm and ready before competing."

# 6️⃣ Ask OpenAI for embeddings (costs < $0.0001 total)
vec1 = client.embeddings.create(input=text1, model="text-embedding-3-small").data[0].embedding
vec2 = client.embeddings.create(input=text2, model="text-embedding-3-small").data[0].embedding

# 7️⃣ Simple cosine similarity function
dot = sum(a*b for a, b in zip(vec1, vec2))                # numerator
mag1 = math.sqrt(sum(a*a for a in vec1))                  # ‖vec1‖
mag2 = math.sqrt(sum(b*b for b in vec2))                  # ‖vec2‖
similarity = dot / (mag1 * mag2)

print(f"Similarity score: {similarity:.3f}")              # 8️⃣ expect ~0.9 (high)
