import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
import pickle

# Configuration
MODEL_PATH = "/home/jazzy/sem-farm/Unet_resnet50/model/optimized_farmland_unet_resnet_150-1024.h5"
INPUT_IMAGE = "/home/jazzy/sem-farm/data/final_farmland_jpg.jpg"  # Path to your large JPG image
OUTPUT_BINARY = "/home/jazzy/sem-farm/Unet_resnet50/output_resnet/output512.jpg"  # Path for merged binary file
CHUNK_SIZE = 512  # Should match your model's expected input size
OVERLAP = 64  # Overlap between chunks to avoid edge artifacts
THRESHOLD = 0.5  # Binary threshold

def load_model():
    """Load the trained U-Net model"""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def split_image(image, chunk_size=256, overlap=32):
    """
    Split large image into overlapping chunks
    Returns:
        chunks: list of image chunks
        positions: list of (x,y) positions for each chunk
        original_shape: shape of original image
    """
    height, width = image.shape[:2]
    chunks = []
    positions = []
    
    # Calculate step size
    step = chunk_size - overlap
    
    # Pad the image if necessary
    pad_x = (step - width % step) if width % step != 0 else 0
    pad_y = (step - height % step) if height % step != 0 else 0
    
    padded_img = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)
    
    # Split into chunks
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Get chunk with overlap consideration
            x1 = max(0, x - overlap//2)
            y1 = max(0, y - overlap//2)
            x2 = min(width, x1 + chunk_size)
            y2 = min(height, y1 + chunk_size)
            
            chunk = padded_img[y1:y2, x1:x2]
            
            # Resize to ensure consistent chunk size
            if chunk.shape[0] != chunk_size or chunk.shape[1] != chunk_size:
                chunk = cv2.resize(chunk, (chunk_size, chunk_size))
            
            chunks.append(chunk)
            positions.append((x1, y1))
    
    return chunks, positions, image.shape

def predict_chunks(model, chunks):
    """Predict binary masks for each chunk"""
    masks = []
    for chunk in chunks:
        # Preprocess
        chunk = chunk / 255.0  # Normalize
        chunk = np.expand_dims(chunk, axis=0)  # Add batch dimension
        
        # Predict
        pred = model.predict(chunk)[0]
        binary_mask = (pred > THRESHOLD).astype(np.uint8)
        masks.append(binary_mask)
    
    return masks

def merge_masks(masks, positions, original_shape, chunk_size=256, overlap=32):
    """Merge predicted chunks back into full-size mask"""
    full_mask = np.zeros(original_shape[:2], dtype=np.float32)
    count_mask = np.zeros(original_shape[:2], dtype=np.float32)

    
    # Create weight matrix for blending
    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)
    
    # Apply linear falloff at edges
    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i-1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i-1] *= (i + 1) / (overlap + 1)
    
    # Blend chunks together
    for mask, (x, y) in zip(masks, positions):
        # Get the actual area to paste (considering image boundaries)
        h, w = mask.shape[:2]
        end_x = min(x + w, original_shape[1])
        end_y = min(y + h, original_shape[0])
        
        # Calculate actual dimensions
        actual_w = end_x - x
        actual_h = end_y - y
        
        # Apply weighted blending
        full_mask[y:end_y, x:end_x] = (
            full_mask[y:end_y, x:end_x] * count_mask[y:end_y, x:end_x] + 
            mask[:actual_h, :actual_w, 0] * weights[:actual_h, :actual_w]
        ) / (count_mask[y:end_y, x:end_x] + weights[:actual_h, :actual_w])
        
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]
    
    # Final threshold
    merged_mask = (full_mask > 0.5).astype(np.uint8)
    return merged_mask

def save_binary_mask(mask, output_path):
    # Multiply by 255 for proper visualization (0 and 255 values)
    mask_visual = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_visual)

def main():
    # Load model
    model = load_model()
    
    # Load and split image
    print("Loading and splitting image...")
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chunks, positions, original_shape = split_image(image, CHUNK_SIZE, OVERLAP)
    
    # Predict masks for chunks
    print(f"Predicting masks for {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)
    
    # Merge masks
    print("Merging predictions...")
    merged_mask = merge_masks(masks, positions, original_shape, CHUNK_SIZE, OVERLAP)
    
    # Save as binary file
    print("Saving merged mask...")
    save_binary_mask(merged_mask, OUTPUT_BINARY)
    
    print(f"Done! Merged mask saved to {OUTPUT_BINARY}")

if __name__ == "__main__":
    main()