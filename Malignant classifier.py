"""
================================================================================
 Malignant vs NotMalignant  —  EfficientNetB0  (single, clean file)
================================================================================

What this file does, top to bottom:
  1. Loads your images from folders
  2. Handles class imbalance (more NotMalignant than Malignant)
  3. Builds an EfficientNetB0 model  (ONE sigmoid output neuron)
  4. Trains in two phases  (frozen base -> fine-tune)
  5. Evaluates AND finds the best decision threshold (the recall fix)
  6. Grad-CAM: shows WHERE the model looks, incl. on the cases it got wrong

HOW TO USE:
  - Edit ONLY the CONFIG block below (your folder paths + a couple of numbers).
  - Then run the file top to bottom.  main() runs everything in order.
  - Anything marked  # <-- EDIT  is something you may want to change.

Folder layout this expects (change paths in CONFIG if yours differ):
    DATA_DIR/
        train/
            Malignant/       *.png / *.jpg ...
            NotMalignant/    *.png / *.jpg ...
        val/
            Malignant/  ...
            NotMalignant/  ...
        test/
            Malignant/  ...
            NotMalignant/  ...

IMPORTANT EfficientNet note:
  EfficientNet in Keras does its OWN normalization inside the model.
  So we feed images as raw 0-255 values and DO NOT rescale to [0,1].
  (If you rescale yourself, the heatmaps and accuracy break. Don't.)
================================================================================
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
#  CONFIG  —  the only part you normally need to touch
# =============================================================================
DATA_DIR   = "/content/drive/MyDrive/your_dataset"   # <-- EDIT: folder with train/val/test
IMG_SIZE   = (224, 224)      # EfficientNetB0's native size — leave as is
BATCH_SIZE = 32              # <-- EDIT if you run out of memory (try 16)
SEED       = 42

# Class order MUST be alphabetical because Keras sorts folder names.
# "Malignant" < "NotMalignant", so: index 0 = Malignant, index 1 = NotMalignant.
CLASS_NAMES = ["Malignant", "NotMalignant"]
POSITIVE_INDEX = 0           # "Malignant" is the class we care about catching

EPOCHS_PHASE1 = 15           # train the new head, base frozen
EPOCHS_PHASE2 = 25           # fine-tune the whole thing, low learning rate
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5             # small! we saw 9e-6 in your logs — same idea

MODEL_OUT = "/content/drive/MyDrive/malignant_effnet.keras"  # <-- EDIT save path

TARGET_RECALL = 0.90         # <-- EDIT: how many malignant cases you want to catch
                             #     0.90 = catch 90%. Higher = safer but more false alarms.


# =============================================================================
#  1. DATA LOADING
# =============================================================================
def load_datasets():
    """Load train/val/test folders into tf.data datasets (labels are 0/1)."""
    common = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",          # integer labels 0/1 (matches sigmoid setup)
        class_names=CLASS_NAMES,   # force the order so index 0 = Malignant
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"), shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"), shuffle=False, **common)

    # Speed: cache + prefetch. Does not change results.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def compute_class_weights(train_dir):
    """More weight to the rarer class (Malignant) so the model stops ignoring it."""
    counts = []
    for name in CLASS_NAMES:
        folder = os.path.join(train_dir, name)
        n = len([f for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        counts.append(n)
        print(f"  {name}: {n} images")

    labels = np.concatenate([[i] * c for i, c in enumerate(counts)])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print("  class_weight =", class_weight)
    return class_weight


# =============================================================================
#  2. MODEL  —  EfficientNetB0 + one sigmoid neuron
# =============================================================================
def build_model():
    """
    Built 'flat' with the functional API so every internal layer (incl. the last
    conv layer) is directly reachable — this is what makes Grad-CAM painless.
    """
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    # Light augmentation (helps the overfitting gap you saw: train 0.94 / val 0.85)
    x = tf.keras.layers.RandomFlip("horizontal", seed=SEED)(inputs)
    x = tf.keras.layers.RandomRotation(0.1, seed=SEED)(x)
    x = tf.keras.layers.RandomZoom(0.1, seed=SEED)(x)

    # weights="imagenet" gives a strong starting point.
    # NOTE: no manual rescaling — EfficientNet normalizes internally.
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False          # Phase 1: freeze the base

    y = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    y = tf.keras.layers.Dropout(0.4)(y)                       # regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(y)  # ONE neuron

    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),  # better under imbalance
        ],
    )


# =============================================================================
#  3. TRAINING  —  two phases
# =============================================================================
def train(model, base, train_ds, val_ds, class_weight):
    # Stop early and keep the best epoch. Monitor PR-AUC (good for imbalance).
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr", mode="max", patience=8,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr", mode="max", factor=0.5, patience=3, verbose=1),
    ]

    print("\n=== PHASE 1: train head, base frozen ===")
    compile_model(model, LR_PHASE1)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1,
              class_weight=class_weight, callbacks=callbacks)

    print("\n=== PHASE 2: fine-tune whole model, low LR ===")
    base.trainable = True                 # unfreeze
    compile_model(model, LR_PHASE2)       # must re-compile after changing trainable
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2,
              class_weight=class_weight, callbacks=callbacks)

    model.save(MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")
    return model


# =============================================================================
#  4. EVALUATION  +  THRESHOLD FIX  (this is what raises malignant recall)
# =============================================================================
def get_probs_and_labels(model, ds):
    """Return the model's malignant-probability and the true label for every image."""
    probs, labels = [], []
    for images, y in ds:
        p = model.predict(images, verbose=0).ravel()   # P(class index 1 = NotMalignant)
        probs.append(p)
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    # Sigmoid outputs P(label==1)=P(NotMalignant). We want P(Malignant):
    malignant_prob = 1.0 - probs
    malignant_true = (labels == POSITIVE_INDEX).astype(int)
    return malignant_prob, malignant_true


def choose_threshold(val_prob, val_true, target_recall):
    """Pick the highest threshold that still catches TARGET_RECALL of malignant cases."""
    prec, rec, thr = precision_recall_curve(val_true, val_prob)
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        print("  (Could not hit target recall; using 0.5)")
        return 0.5
    best = thr[ok[-1]]
    print(f"  threshold for recall>={target_recall}: {best:.3f} "
          f"(precision there ≈ {prec[ok[-1]]:.2f})")
    return float(best)


def evaluate(model, val_ds, test_ds):
    print("\n=== EVALUATION ===")
    val_prob, val_true = get_probs_and_labels(model, val_ds)
    test_prob, test_true = get_probs_and_labels(model, test_ds)

    print("\nAt default threshold 0.5:")
    print(classification_report(
        test_true, (test_prob >= 0.5).astype(int),
        target_names=["NotMalignant", "Malignant"], digits=3))

    print("Choosing threshold on VALIDATION (never on test):")
    thr = choose_threshold(val_prob, val_true, TARGET_RECALL)

    print(f"\nAt tuned threshold {thr:.3f}:")
    print(classification_report(
        test_true, (test_prob >= thr).astype(int),
        target_names=["NotMalignant", "Malignant"], digits=3))
    return thr, test_prob, test_true


# =============================================================================
#  5. GRAD-CAM  —  shows where the model looks
# =============================================================================
def find_last_conv_layer(model):
    """Auto-find the last layer with a 2D feature map. Works nested or flat."""
    for layer in reversed(model.layers):
        # a conv feature map has shape (batch, H, W, channels) -> 4 dims
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 4D (conv) layer found for Grad-CAM.")


def compute_gradcam(model, img_batch, last_conv_name):
    """img_batch shape: (1, H, W, 3), raw 0-255. Returns a 0..1 heatmap."""
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(last_conv_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        # single sigmoid: preds[:,0] is P(NotMalignant); malignant score = 1 - that
        malignant_score = 1.0 - preds[:, 0]

    grads = tape.gradient(malignant_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))         # importance per channel
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def show_gradcam(model, image, last_conv_name, title=""):
    """image: a single (H, W, 3) array in 0-255. Displays original + overlay."""
    batch = np.expand_dims(image, 0).astype("float32")
    heatmap = compute_gradcam(model, batch, last_conv_name)

    heatmap = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image.astype("uint8")); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(image.astype("uint8"))
    ax[1].imshow(heatmap, cmap="jet", alpha=0.45)
    ax[1].set_title("Grad-CAM: where model looks"); ax[1].axis("off")
    fig.suptitle(title)
    plt.tight_layout(); plt.show()


def gradcam_on_mistakes(model, test_ds, threshold, last_conv_name, max_show=6):
    """Show Grad-CAM specifically on MISSED malignant cases (the false negatives)."""
    print("\n=== Grad-CAM on missed malignant cases ===")
    last_conv_name = last_conv_name or find_last_conv_layer(model)
    shown = 0
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        malignant_prob = 1.0 - probs
        malignant_true = (labels.numpy() == POSITIVE_INDEX).astype(int)
        predicted = (malignant_prob >= threshold).astype(int)

        for i in range(len(images)):
            missed = (malignant_true[i] == 1 and predicted[i] == 0)  # false negative
            if missed:
                show_gradcam(model, images[i].numpy(), last_conv_name,
                             title=f"MISSED malignant (p={malignant_prob[i]:.2f})")
                shown += 1
                if shown >= max_show:
                    return
    if shown == 0:
        print("  No missed malignant cases at this threshold — nice.")


# =============================================================================
#  MAIN  —  runs everything in order
# =============================================================================
def main():
    print(">> Loading data")
    train_ds, val_ds, test_ds = load_datasets()

    print(">> Class balance")
    class_weight = compute_class_weights(os.path.join(DATA_DIR, "train"))

    print(">> Building model")
    model, base = build_model()

    print(">> Training")
    model = train(model, base, train_ds, val_ds, class_weight)

    print(">> Evaluating + tuning threshold")
    threshold, _, _ = evaluate(model, val_ds, test_ds)

    print(">> Grad-CAM on mistakes")
    last_conv = find_last_conv_layer(model)
    print("   last conv layer:", last_conv)
    gradcam_on_mistakes(model, test_ds, threshold, last_conv)


if __name__ == "__main__":
    main()
