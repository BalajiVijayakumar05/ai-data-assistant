import faiss
import numpy as np
from openai import OpenAI
from utils import load_dataframe, row_to_text
from config import OPENAI_API_KEY, MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

df = load_dataframe()
texts = df.apply(row_to_text, axis=1).tolist()

# Generate embeddings
embeddings = []
for t in texts:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=t
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "../embeddings/data_index.faiss")
df.to_pickle("../embeddings/data.pkl")

print("✅ Embeddings created and index stored!")