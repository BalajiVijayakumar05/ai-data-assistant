import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Load index + data
index = faiss.read_index("../embeddings/data_index.faiss")
df = pd.read_pickle("../embeddings/data.pkl")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32")


def ask_data_assistant(question):
    # 1) Convert question to embedding
    q_emb = get_embedding(question)
    q_emb = np.expand_dims(q_emb, axis=0)

    # 2) Retrieve top 3 rows
    distances, indices = index.search(q_emb, 3)
    retrieved_rows = df.iloc[indices[0]]

    # 3) Build context
    context = "\n".join(retrieved_rows.apply(lambda r: str(r.to_dict()), axis=1))

    # 4) Ask LLM
    prompt = f"""
You are an AI Data Assistant. Use ONLY the data below to answer.

Data:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question about the dataset: ")
        print(ask_data_assistant(q))