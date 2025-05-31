import os, cv2, json, numpy as np, tensorflow as tf
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-4
THRESHOLD = 0.5

# --- Paths ---
DATASET_PATH = "/home/jazzy/sem-farm/datasets"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")
ANNOT_PATHS = {
    "train": os.path.join(TRAIN_DIR, "_annotations.coco.json"),
    "valid": os.path.join(VALID_DIR, "_annotations.coco.json"),
    "test": os.path.join(TEST_DIR, "_annotations.coco.json"),
}

# --- Data Loader ---
def load_data(image_dir, annotation_path):
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()
    X, Y = [], []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        Y.append(np.expand_dims(mask, -1))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.uint8)

# --- Model: U-Net with ResNet50 Encoder ---
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(input_shape)
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    skip_connections = [
        backbone.get_layer("conv1_relu").output,         # 128
        backbone.get_layer("conv2_block3_out").output,   # 64
        backbone.get_layer("conv3_block4_out").output,   # 32
        backbone.get_layer("conv4_block6_out").output,   # 16
    ]
    encoder_output = backbone.get_layer("conv5_block3_out").output  # 8

    x = encoder_output
    for i, skip in reversed(list(enumerate(skip_connections))):
        x = Conv2DTranspose(512 // (2 ** i), (2, 2), strides=2, padding="same")(x)
        x = concatenate([x, skip])
        x = Conv2D(512 // (2 ** i), (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.2)(x)

    x = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)
    return Model(inputs, outputs)

# --- Loss ---
def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice + bce

# --- Load Data ---
X_train, Y_train = load_data(TRAIN_DIR, ANNOT_PATHS["train"])
X_val, Y_val = load_data(VALID_DIR, ANNOT_PATHS["valid"])
X_test, Y_test = load_data(TEST_DIR, ANNOT_PATHS["test"])

# --- Train Model ---
model = build_unet()
model.compile(optimizer=Adam(LR), loss=dice_bce_loss, metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("/home/jazzy/sem-farm/Unet_resnet50/model/optimized_farmland_unet_resnet_150-1024.h5")

# --- Evaluate Model ---
def evaluate(model, X, Y, threshold=THRESHOLD):
    preds = (model.predict(X) > threshold).astype(np.uint8)
    y_true = Y.flatten()
    y_pred = preds.flatten()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    f2 = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0
    iou = np.sum(y_true * y_pred) / np.sum((y_true + y_pred) > 0)
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | IoU: {iou:.4f}")

evaluate(model, X_test, Y_test)

# --- Visualization ---
def visualize(img, true_mask=None):
    pred = (model.predict(np.expand_dims(img, 0))[0, :, :, 0] > THRESHOLD).astype(np.uint8)
    fig, ax = plt.subplots(1, 3 if true_mask is not None else 2, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    if true_mask is not None:
        ax[1].imshow(true_mask.squeeze(), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred, cmap='gray')
        ax[2].set_title("Prediction")
    else:
        ax[1].imshow(pred, cmap='gray')
        ax[1].set_title("Prediction")
    plt.show()

# Example:
visualize(X_test[0], Y_test[0])
