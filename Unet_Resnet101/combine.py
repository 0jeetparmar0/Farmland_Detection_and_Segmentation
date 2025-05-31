import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet101

# --- Configuration ---
MODEL_PATH = "/home/jazzy/sem-farm/unet_best_resnet101.h5"
INPUT_IMAGE = "/home/jazzy/sem-farm/data/final_farmland_jpg.jpg"
OUTPUT_BINARY = "/home/jazzy/sem-farm/Unet_resnet50/output_resnet/output_resnet101.jpg"
CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5

# --- Model Setup ---
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    backbone = ResNet101(include_top=False, weights="imagenet", input_tensor=inputs)

    skip_connections = [
        backbone.get_layer("conv1_relu").output,
        backbone.get_layer("conv2_block3_out").output,
        backbone.get_layer("conv3_block4_out").output,
        backbone.get_layer("conv4_block23_out").output,
    ]
    encoder_output = backbone.get_layer("conv5_block3_out").output

    x = encoder_output
    for i, skip in reversed(list(enumerate(skip_connections))):
        filters = 256 // (2 ** i)
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        x = concatenate([x, skip])
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.2)(x)

    x = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    return Model(inputs, outputs)

def load_model():
    model = build_unet(input_shape=(CHUNK_SIZE, CHUNK_SIZE, 3))
    model.load_weights(MODEL_PATH)
    return model

# --- Image Split ---
def split_image(image, chunk_size=256, overlap=32):
    height, width = image.shape[:2]
    step = chunk_size - overlap
    pad_x = (step - width % step) if width % step != 0 else 0
    pad_y = (step - height % step) if height % step != 0 else 0

    padded_img = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)
    chunks, positions = [], []

    for y in range(0, height, step):
        for x in range(0, width, step):
            x1 = max(0, x - overlap // 2)
            y1 = max(0, y - overlap // 2)
            x2 = x1 + chunk_size
            y2 = y1 + chunk_size
            chunk = padded_img[y1:y2, x1:x2]
            if chunk.shape[:2] != (chunk_size, chunk_size):
                chunk = cv2.resize(chunk, (chunk_size, chunk_size))
            chunks.append(chunk)
            positions.append((x1, y1))

    return chunks, positions, image.shape

# --- Prediction ---
def predict_chunks(model, chunks):
    masks = []
    for chunk in chunks:
        img = chunk / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)[0, :, :, 0]
        binary = (pred > THRESHOLD).astype(np.uint8)
        masks.append(binary)
    return masks

# --- Merge Prediction ---
def merge_masks(masks, positions, original_shape, chunk_size=256, overlap=32):
    full_mask = np.zeros(original_shape[:2], dtype=np.float32)
    count_mask = np.zeros(original_shape[:2], dtype=np.float32)
    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)

    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i - 1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i - 1] *= (i + 1) / (overlap + 1)

    for mask, (x, y) in zip(masks, positions):
        h, w = mask.shape
        end_x = min(x + w, original_shape[1])
        end_y = min(y + h, original_shape[0])
        actual_w = end_x - x
        actual_h = end_y - y

        full_mask[y:end_y, x:end_x] = (
            full_mask[y:end_y, x:end_x] * count_mask[y:end_y, x:end_x] +
            mask[:actual_h, :actual_w] * weights[:actual_h, :actual_w]
        ) / (count_mask[y:end_y, x:end_x] + weights[:actual_h, :actual_w])
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]

    return (full_mask > 0.5).astype(np.uint8)

# --- Save ---
def save_binary_mask(mask, output_path):
    cv2.imwrite(output_path, (mask * 255).astype(np.uint8))

# --- Main Flow ---
def main():
    print("Loading model...")
    model = load_model()

    print("Reading image...")
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Splitting image...")
    chunks, positions, original_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print(f"Predicting {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)

    print("Merging predictions...")
    merged = merge_masks(masks, positions, original_shape, CHUNK_SIZE, OVERLAP)

    print("Saving result...")
    save_binary_mask(merged, OUTPUT_BINARY)

    print(f"✅ Done! Mask saved to {OUTPUT_BINARY}")

if __name__ == "__main__":
    main()
