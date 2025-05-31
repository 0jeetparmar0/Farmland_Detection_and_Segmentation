import os
import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- Suppress oneDNN Warning ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Set GPU Memory Growth Before Any TF Operations ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Memory growth enabled for GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Warning: Could not set memory growth: {e}")
else:
    print("No GPU detected, running on CPU")

# --- Enable mixed precision and XLA ---
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 300
BASE_LR = 1e-4 
WARMUP_EPOCHS = 5
NUM_CLASSES = 1

DATASET_PATH = "/home/jazzy/sem-farm/datasets"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")
ANNOT_PATHS = {
    "train": os.path.join(TRAIN_DIR, "_annotations.coco.json"),
    "valid": os.path.join(VALID_DIR, "_annotations.coco.json"),
    "test": os.path.join(TEST_DIR, "_annotations.coco.json"),
}

# --- Data Augmentation ---
def augment(image, mask):
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, mask

# --- Data Loader ---
def load_data(image_dir, annotation_path):
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()

    def generator():
        for img_id in image_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)

            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            yield img, np.expand_dims(mask, -1)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.uint8),
        output_shapes=([IMG_SIZE, IMG_SIZE, 3], [IMG_SIZE, IMG_SIZE, 1])
    )

    return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Attention Gate ---
def attention_gate(input_x, skip, filters):
    g = Conv2D(filters, (1, 1), padding="same")(input_x)
    x = Conv2D(filters, (1, 1), padding="same")(skip)
    x = tf.keras.layers.Add()([g, x])
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return tf.keras.layers.Multiply()([skip, x])

# --- Model Definition ---
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), freeze_encoder=True):
    inputs = Input(input_shape)
    backbone = ResNet101(include_top=False, weights="imagenet", input_tensor=inputs)

    if freeze_encoder:
        for layer in backbone.layers[:100]:
            layer.trainable = False

    skip_connections = [
        backbone.get_layer("conv1_relu").output,
        backbone.get_layer("conv2_block3_out").output,
        backbone.get_layer("conv3_block4_out").output,
        backbone.get_layer("conv4_block23_out").output,
    ]
    encoder_output = backbone.get_layer("conv5_block3_out").output

    x = encoder_output
    for i, skip in reversed(list(enumerate(skip_connections))):
        filters = 128 // (2 ** i)
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        skip = attention_gate(x, skip, filters)
        x = concatenate([x, skip])
        x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.3)(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(x)
    outputs = Conv2D(1, (1, 1), activation="sigmoid", dtype="float32")(x)

    return Model(inputs, outputs)

# --- Loss Functions ---
@tf.function
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    focal_factor = tf.pow(1 - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1 - y_true)
    return tf.reduce_mean(alpha * focal_factor * bce)

@tf.function
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice_coeff + bce + 0.5 * focal

# --- Learning Rate Schedule ---
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base, total_steps, warmup_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.learning_rate_base * (step / self.warmup_steps)
        cosine_lr = self.learning_rate_base * 0.5 * (
            1 + tf.math.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

# --- Training Setup ---
train_dataset = load_data(TRAIN_DIR, ANNOT_PATHS["train"])
val_dataset = load_data(VALID_DIR, ANNOT_PATHS["valid"])
test_dataset = load_data(TEST_DIR, ANNOT_PATHS["test"]).unbatch().batch(BATCH_SIZE)

model = build_unet()
total_steps = (len(COCO(ANNOT_PATHS["train"]).getImgIds()) // BATCH_SIZE) * EPOCHS
warmup_steps = (len(COCO(ANNOT_PATHS["train"]).getImgIds()) // BATCH_SIZE) * WARMUP_EPOCHS
lr_schedule = WarmUpCosineDecay(BASE_LR, total_steps, warmup_steps)
optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

model.compile(optimizer=optimizer, loss=combined_loss, metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint("unet_best_resnet300.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stop_cb = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_schedule_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb, lr_schedule_cb]
)

# --- Evaluation ---
def evaluate(model, dataset, threshold=0.5):
    preds, y_true = [], []
    for x, y in dataset:
        pred = (model.predict(x) > threshold).astype(np.uint8)
        preds.append(pred.flatten())
        y_true.append(y.numpy().flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(preds)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    f2 = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0
    iou = np.sum(y_true * y_pred) / np.sum((y_true + y_pred) > 0)
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | IoU: {iou:.4f}")

evaluate(model, test_dataset)

# --- Visualization ---
def visualize(img, true_mask=None):
    pred = (model.predict(np.expand_dims(img, 0))[0, :, :, 0] > 0.5).astype(np.uint8)
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

for x, y in test_dataset.take(1):
    visualize(x[0], y[0])