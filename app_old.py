import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import json
import os

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Streamlit app layout
st.title("CLIP Image Embedding Generator")

# File uploader for single image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    # Preprocess the image and get CLIP embedding
    image = Image.open(uploaded_file)
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    embedding_array = embedding.cpu().numpy()

    # Convert embedding to 1D list for JSON compatibility
    embedding_list = embedding_array.flatten().tolist()

    # Load multi-image embeddings and filenames
    embeddings_path = "/Users/nick_leong/Library/CloudStorage/GoogleDrive-nick@mooreplasticresearch.org/My Drive/embeddings/clip_embeddings.json"  # Update this path if needed
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "r") as f:
            multi_data = json.load(f)
            file_names = multi_data["file_names"]
            embeddings_array = np.array(multi_data["embeddings"])
            # Assuming images are stored in a directory and file names are relative paths
            image_directory = "path/to/your/images"  # Update this path if needed

        # Ensure dimensions match
        embedding_dim = embedding_array.shape[1]
        if embeddings_array.shape[1] != embedding_dim:
            st.error(f"Dimension mismatch: single embedding dimension ({embedding_dim}) does not match multi-image embeddings dimension ({embeddings_array.shape[1]}).")
            st.stop()

        # Normalize the single embedding
        embedding_single = np.array(embedding_list)
        if np.linalg.norm(embedding_single) != 0:
            embedding_single = embedding_single / np.linalg.norm(embedding_single)
        else:
            st.error("The single embedding is a zero vector and cannot be normalized.")
            st.stop()

        # Normalize all multi embeddings
        if embeddings_array.size > 0:
            norms = np.linalg.norm(embeddings_array, axis=1)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms[:, np.newaxis]
        else:
            st.error("The multi-image embeddings array is empty.")
            st.stop()

        # Calculate similarities
        similarities = np.dot(embeddings_array, embedding_single)

        # Get the indices of the top 10 most similar images
        top_k = 10
        top_indices = np.argsort(similarities)[-top_k:]  # Get the indices of the top 10 similarities

        # Display the top 10 most similar images and their similarity scores
        st.write(f"Top {top_k} most similar images:")
        for index in reversed(top_indices):  # reversed to get the highest similarity first
            # Load the image corresponding to the filename
            img_path = os.path.join(image_directory, file_names[index])
            try:
                img = Image.open(img_path)
                st.image(img, caption=f"Filename: {file_names[index]}")
                st.write(f"Similarity score: {similarities[index]}")
            except FileNotFoundError:
                st.write(f"Image not found: {file_names[index]}")
    else:
        st.error("Embeddings file not found.")
