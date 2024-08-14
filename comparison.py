import numpy as np
import json

# Load multi-image embeddings and filenames
with open("data/clip_embeddings.json", "r") as f:
    multi_data = json.load(f)
    file_names = multi_data["file_names"]
    embeddings_array = np.array(multi_data["embeddings"])

# Load single image embedding and filename
with open("data/single_image_embedding.json", "r") as f:
    single_data = json.load(f)
    single_image_path = single_data["file_name"]
    embedding_single = np.array(single_data["embedding"])

# Ensure the single embedding is reshaped correctly
embedding_single = embedding_single.flatten()

# Normalize the single embedding
if np.linalg.norm(embedding_single) != 0:
    embedding_single = embedding_single / np.linalg.norm(embedding_single)
else:
    raise ValueError("The single embedding is a zero vector and cannot be normalized.")

# Normalize all multi embeddings
if embeddings_array.size > 0 and embeddings_array.shape[1] == embedding_single.shape[0]:
    norms = np.linalg.norm(embeddings_array, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms[:, np.newaxis]
else:
    raise ValueError("Mismatched dimensions between single embedding and multi embeddings, or empty embeddings.")

# Calculate similarities
similarities = np.dot(embeddings_array, embedding_single)

# Get the indices of the top 10 most similar images
top_k = 10
top_indices = np.argsort(similarities)[-top_k:]  # Get the indices of the top 10 similarities

# Print the top 10 most similar images and their similarity scores
print(f"Top {top_k} most similar images:")
for index in reversed(top_indices):  # reversed to get the highest similarity first
    print(f"Filename: {file_names[index]}")
    print(f"Similarity score: {similarities[index]}")
    print()
