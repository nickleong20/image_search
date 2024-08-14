import os
from PIL import Image
import clip
import torch
import numpy as np
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing images
root_dir = "/Users/nick_leong/Library/CloudStorage/GoogleDrive-nick@mooreplasticresearch.org/My Drive/images"

# Initialize lists to store embeddings and filenames
embeddings = []
file_names = []

# Traverse the directory and process images
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(root, file)
            print(f"Processing {image_path}")
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            embeddings.append(embedding.cpu().numpy())
            file_names.append(image_path)  # Store file path for each image

# Convert embeddings to a NumPy array
embeddings_array = np.vstack(embeddings)

# Create a dictionary to store embeddings and filenames
embeddings_dict = {
    "file_names": file_names,
    "embeddings": embeddings_array.tolist()  # Convert to list for JSON serialization
}

# Save the embeddings and filenames as a .json file
output_file = "data/clip_embeddings.json"
with open(output_file, "w") as f:
    json.dump(embeddings_dict, f)

print(f"Multi-image embeddings and filenames saved successfully to {output_file}!")
