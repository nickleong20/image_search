import os
from PIL import Image
import clip
import torch
import numpy as np
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Path for the single image
single_image_path = "/Users/nick_leong/Library/CloudStorage/GoogleDrive-nick@mooreplasticresearch.org/.shortcut-targets-by-id/1fDLUwBkwKUAtqFv8S4m-vm8z0Q4gcTSK/test_dataset/hard plastic/PAL_08_8.JPG"

# Preprocess the single image and get CLIP embedding
single_image = preprocess(Image.open(single_image_path)).unsqueeze(0).to(device)
with torch.no_grad():
    single_embedding = model.encode_image(single_image)
single_embedding_array = single_embedding.cpu().numpy()

# Create a dictionary to store the single image embedding and filename
single_embedding_dict = {
    "file_name": single_image_path,
    "embedding": single_embedding_array.tolist()  # Convert to list for JSON serialization
}

# Save the single image embedding and filename as a .json file
single_output_file = "/Users/nick_leong/Library/CloudStorage/GoogleDrive-nick@mooreplasticresearch.org/My Drive/embeddings/single_image_embedding.json"
with open(single_output_file, "w") as f:
    json.dump(single_embedding_dict, f)

print(f"Single image embedding and filename saved successfully to {single_output_file}!")

