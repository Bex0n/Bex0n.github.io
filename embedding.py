import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('vocab-transformers/dense_encoder-msmarco-distilbert-word2vec256k')

with open("20k.txt") as f:
    words = f.read().splitlines()[:30000]

embeddings = model.encode(words)

# Save embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    pairs = {word: emb.tolist() for word, emb in zip(words, embeddings)}
    json.dump(pairs, f)

# Hardcoded word
target_word = 'example'
if target_word not in words:
    raise ValueError(f"The target word '{target_word}' is not in the word list.")

target_embedding = model.encode([target_word])[0]

# Compute similarities and rank the words
similarities = []
for word, embedding in zip(words, embeddings):
    similarity = np.dot(target_embedding, embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(embedding))
    similarities.append((word, similarity))

# Sort words by similarity in descending order
similarities.sort(key=lambda x: x[1], reverse=True)

# Create ranking.json with {word: position_in_ranking}
ranking = {word: rank + 1 for rank, (word, _) in enumerate(similarities)}

# Save ranking to a JSON file
with open('ranking.json', 'w') as f:
    json.dump(ranking, f)
