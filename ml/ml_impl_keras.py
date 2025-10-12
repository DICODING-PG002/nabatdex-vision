import os
import math
from typing import Any, Dict, List, Tuple
import zipfile
import shutil

import cv2
import numpy as np
import tensorflow as tf

from .ml_core import (
    DataSplits,
    register_data_source,
    register_augmenter,
    register_model_builder,
    register_trainer,
    register_evaluator,
    register_exporter,
)


# -------------------------
# Helpers
# -------------------------

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _read_image(path: str, size: Tuple[int, int], color_mode: str) -> np.ndarray:
    if color_mode == "grayscale":
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img[..., None]
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _load_dir_images(
    dir_path: str,
    classes: List[str],
    img_size: Tuple[int, int],
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(dir_path, cls)
        if not os.path.isdir(cls_dir):
            # Skip missing class dir in this split
            continue
        for name in os.listdir(cls_dir):
            p = os.path.join(cls_dir, name)
            # Basic image file filter
            if not os.path.isfile(p):
                continue
            if not (name.lower().endswith(".png") or name.lower().endswith(".jpg") or name.lower().endswith(".jpeg") or name.lower().endswith(".bmp")):
                continue
            img = _read_image(p, img_size, color_mode)
            X.append(img)
            y.append(idx)
    if len(X) == 0:
        return np.empty((0, img_size[1], img_size[0], 1 if color_mode == "grayscale" else 3), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)
    return X_arr, y_arr


# -------------------------
# Data Sources
# -------------------------

def data_from_directory(cfg: Dict[str, Any]) -> DataSplits:
    train_dir: str = cfg.get("train_dir", "data/train")
    val_dir: str = cfg.get("val_dir", "data/val")
    test_dir: str = cfg.get("test_dir", "data/test")
    img_height: int = int(cfg.get("img_height", 28))
    img_width: int = int(cfg.get("img_width", 28))
    color_mode: str = cfg.get("color_mode", "grayscale")
    normalize: bool = bool(cfg.get("normalize", True))

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    classes = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    if len(classes) == 0:
        raise ValueError(f"No class subdirectories found under {train_dir}")

    size = (img_width, img_height)
    X_train, y_train = _load_dir_images(train_dir, classes, size, color_mode)
    
    # Create empty arrays with proper dtype
    empty_X_shape = (0, img_height, img_width, 1 if color_mode == "grayscale" else 3)
    empty_y_shape = (0,)
    
    if os.path.isdir(val_dir):
        X_val, y_val = _load_dir_images(val_dir, classes, size, color_mode)
    else:
        X_val, y_val = np.empty(empty_X_shape, dtype=X_train.dtype), np.empty(empty_y_shape, dtype=np.int64)
    
    if os.path.isdir(test_dir):
        X_test, y_test = _load_dir_images(test_dir, classes, size, color_mode)
    else:
        X_test, y_test = np.empty(empty_X_shape, dtype=X_train.dtype), np.empty(empty_y_shape, dtype=np.int64)

    if normalize and X_train.size:
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0 if X_val.size else X_val
        X_test = X_test.astype(np.float32) / 255.0 if X_test.size else X_test

    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def data_keras_mnist(cfg: Dict[str, Any]) -> DataSplits:
    from tensorflow.keras.datasets import mnist
    from sklearn.model_selection import train_test_split

    val_ratio: float = float(cfg.get("val_ratio", 0.1))
    random_state: int = int(cfg.get("random_state", 42))

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize and add channel dim
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )

    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def data_kaggle_potato_disease(cfg: Dict[str, Any]) -> DataSplits:
    """
    Load potato disease dataset from Kaggle.
    Downloads only Potato classes: Early_blight, Late_blight, healthy
    
    Config params:
        - cache_dir: Directory to cache dataset (default: "data/potato_disease")
        - img_height: Image height (default: 256)
        - img_width: Image width (default: 256)
        - color_mode: "rgb" or "grayscale" (default: "rgb")
        - normalize: Normalize to [0,1] (default: True)
        - val_split: Validation split ratio (default: 0.15)
        - test_split: Test split ratio (default: 0.15)
        - random_state: Random seed (default: 42)
        - force_download: Force re-download even if cached (default: False)
    """
    from sklearn.model_selection import train_test_split
    
    cache_dir: str = cfg.get("cache_dir", "data/potato_disease")
    img_height: int = int(cfg.get("img_height", 256))
    img_width: int = int(cfg.get("img_width", 256))
    color_mode: str = cfg.get("color_mode", "rgb")
    normalize: bool = bool(cfg.get("normalize", True))
    val_split: float = float(cfg.get("val_split", 0.15))
    test_split: float = float(cfg.get("test_split", 0.15))
    random_state: int = int(cfg.get("random_state", 42))
    force_download: bool = bool(cfg.get("force_download", False))
    subset_fraction: float = float(cfg.get("subset_fraction", 1.0))
    max_per_class: int = int(cfg.get("max_images_per_class", 0))
    rng = np.random.RandomState(random_state)
    
    # Ensure cache directory exists
    _ensure_dir(cache_dir)
    
    # Define potato classes we want to extract
    potato_classes = [
        "Potato___Early_blight",
        "Potato___Late_blight", 
        "Potato___healthy"
    ]
    
    # Check if data is already extracted
    data_ready = all(
        os.path.isdir(os.path.join(cache_dir, cls)) and 
        len(os.listdir(os.path.join(cache_dir, cls))) > 0
        for cls in potato_classes
    )
    
    if not data_ready or force_download:
        print("Downloading Kaggle dataset...")
        try:
            import kagglehub
            
            # Download the dataset
            dataset_path = kagglehub.dataset_download("karagwaanntreasure/plant-disease-detection")
            print(f"Dataset downloaded to: {dataset_path}")
            
            # Find the main directory with the plant data
            # The dataset might be in different structures, so we need to search for it
            plant_dir = None
            for root, dirs, files in os.walk(dataset_path):
                # Look for a directory that contains our potato classes
                if any(d.startswith("Potato___") for d in dirs):
                    plant_dir = root
                    break
            
            if plant_dir is None:
                raise ValueError(f"Could not find plant disease data in {dataset_path}")
            
            # Extract only potato classes
            print(f"Extracting potato classes from {plant_dir}...")
            for cls in potato_classes:
                src_dir = os.path.join(plant_dir, cls)
                dst_dir = os.path.join(cache_dir, cls)
                
                if not os.path.isdir(src_dir):
                    print(f"Warning: Class directory not found: {src_dir}")
                    continue
                
                # Clean destination if it exists
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                
                # Copy the directory
                shutil.copytree(src_dir, dst_dir)
                num_images = len([f for f in os.listdir(dst_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {cls}: {num_images} images")
                
        except ImportError:
            raise ImportError(
                "kagglehub is required to download the dataset. "
                "Install it with: pip install kagglehub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download/extract dataset: {e}")
    else:
        print(f"Using cached dataset from {cache_dir}")
    
    # Load all images
    print("Loading images...")
    size = (img_width, img_height)
    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    
    class_names = sorted(potato_classes)  # Ensure consistent ordering
    
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(cache_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Skipping missing class directory: {cls_dir}")
            continue
        names = [n for n in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, n)) and (n.lower().endswith(".png") or n.lower().endswith(".jpg") or n.lower().endswith(".jpeg"))]
        if subset_fraction < 1.0:
            k = max(1, int(len(names) * subset_fraction))
            rng.shuffle(names)
            names = names[:k]
        if max_per_class > 0:
            names = names[:max_per_class]
        for name in names:
            p = os.path.join(cls_dir, name)
            try:
                img = _read_image(p, size, color_mode)
                X_all.append(img)
                y_all.append(idx)
            except Exception as e:
                print(f"Warning: Failed to read {p}: {e}")
                continue
    
    if len(X_all) == 0:
        raise ValueError("No images loaded. Please check the dataset.")
    
    X_all = np.stack(X_all, axis=0)
    y_all = np.asarray(y_all, dtype=np.int64)
    
    print(f"Loaded {len(X_all)} images across {len(class_names)} classes")
    
    # Split into train, validation, and test sets
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, 
        test_size=test_split, 
        random_state=random_state, 
        stratify=y_all
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    # Normalize if requested
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class mapping: {dict(enumerate(class_names))}")
    
    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)


register_data_source("from_directory", data_from_directory)
register_data_source("keras_mnist", data_keras_mnist)
register_data_source("kaggle_potato_disease", data_kaggle_potato_disease)


# -------------------------
# Augmenters
# -------------------------

def augmenter_identity(splits: DataSplits, cfg: Dict[str, Any]) -> DataSplits:
    return splits


def _random_rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, np.random.uniform(-degrees, degrees), 1.0)
    if len(img.shape) == 2 or img.shape[-1] == 1:
        # Handle 2D grayscale or 3D with single channel
        img_2d = img.squeeze() if len(img.shape) == 3 else img
        rotated = cv2.warpAffine(img_2d, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rotated[..., None] if len(img.shape) == 3 else rotated
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _random_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    # delta in [0, 1]; apply factor in [1-delta, 1+delta]
    factor = 1.0 + np.random.uniform(-delta, delta)
    out = np.clip(img * factor, 0.0, 1.0)
    return out.astype(img.dtype)


def _random_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    if len(img.shape) == 2 or img.shape[-1] == 1:
        # Handle 2D grayscale or 3D with single channel
        img_2d = img.squeeze() if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(img_2d, (k, k), 0)
        return blurred[..., None] if len(img.shape) == 3 else blurred
    return cv2.GaussianBlur(img, (k, k), 0)


def augmenter_mnist_basic(splits: DataSplits, cfg: Dict[str, Any]) -> DataSplits:
    p: float = float(cfg.get("p", 0.25))
    max_rotate: float = float(cfg.get("max_rotate", 20.0))
    brightness_delta: float = float(cfg.get("brightness_delta", 0.1))
    blur_kernel: int = int(cfg.get("blur_kernel", 3))

    X, y = splits.X_train, splits.y_train
    if X.size == 0:
        return splits

    aug_X: List[np.ndarray] = []
    aug_y: List[int] = []
    for i in range(X.shape[0]):
        x = X[i]
        label = int(y[i])
        if np.random.rand() < p:
            choice = np.random.choice(["rotate", "brightness", "blur"])
            if choice == "rotate":
                x_aug = _random_rotate(x, max_rotate)
            elif choice == "brightness":
                x_aug = _random_brightness(x, brightness_delta)
            else:
                x_aug = _random_blur(x, blur_kernel)
            aug_X.append(x_aug)
            aug_y.append(label)

    if len(aug_X) == 0:
        return splits

    X_train = np.concatenate([X, np.stack(aug_X)], axis=0)
    y_train = np.concatenate([y, np.asarray(aug_y, dtype=y.dtype)], axis=0)
    return DataSplits(X_train, y_train, splits.X_val, splits.y_val, splits.X_test, splits.y_test)


register_augmenter("identity", augmenter_identity)
register_augmenter("mnist_basic", augmenter_mnist_basic)


# -------------------------
# Model Builders
# -------------------------

def build_sequential_cnn_mnist(
    input_shape: Tuple[int, int, int], num_classes: int, cfg: Dict[str, Any]
) -> tf.keras.Model:
    from tensorflow.keras import layers, models, optimizers

    filters: List[int] = list(cfg.get("filters", [32, 32, 32]))
    kernels_raw = cfg.get("kernels", [(3, 3), (4, 4), (7, 7)])
    kernels: List[Tuple[int, int]] = [tuple(k) if isinstance(k, (list, tuple)) else (k, k) for k in kernels_raw]
    pool_size: Tuple[int, int] = tuple(cfg.get("pool_size", (2, 2)))
    dense_units: List[int] = list(cfg.get("dense_units", [128, 64]))
    dropout_rates: List[float] = list(cfg.get("dropout_rates", [0.5, 0.3]))
    optimizer_name: str = str(cfg.get("optimizer", "RMSprop"))
    learning_rate: float = float(cfg.get("learning_rate", 0.001))

    model = models.Sequential(name="mnist_sequential_cnn")
    model.add(layers.Input(shape=input_shape, name="input_layer"))

    for i, (f, k) in enumerate(zip(filters, kernels)):
        model.add(layers.Conv2D(f, k, padding="same", activation="relu", name=f"conv2d_{i+1}"))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.MaxPooling2D(pool_size, name=f"pool_{i+1}"))

    model.add(layers.Flatten(name="flatten"))
    for j, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{j+1}"))
        if j < len(dropout_rates):
            model.add(layers.Dropout(dropout_rates[j], name=f"dropout_{j+1}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    opt: tf.keras.optimizers.Optimizer
    if optimizer_name.lower() == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    else:
        # Get optimizer and configure learning rate
        opt = optimizers.get(optimizer_name)
        if hasattr(opt, 'learning_rate'):
            opt.learning_rate = learning_rate

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


register_model_builder("sequential_cnn_mnist", build_sequential_cnn_mnist)


def build_mobilenet_v2_transfer(
    input_shape: Tuple[int, int, int], num_classes: int, cfg: Dict[str, Any]
) -> tf.keras.Model:
    from tensorflow.keras import layers, models, optimizers, mixed_precision as mp

    if bool(cfg.get("mixed_precision", False)):
        try:
            mp.set_global_policy("mixed_float16")
        except Exception:
            pass

    base_input_shape = (input_shape[0], input_shape[1], 3)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if input_shape[-1] == 3 else None,
        input_shape=base_input_shape,
    )
    if bool(cfg.get("freeze_base", True)):
        base.trainable = False

    drop = float(cfg.get("dropout", 0.2))
    dense_units = int(cfg.get("dense_units", 128))
    opt_name = str(cfg.get("optimizer", "Adam"))
    lr = float(cfg.get("learning_rate", 0.0005))

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if input_shape[-1] != 3:
        x = tf.keras.layers.Conv2D(3, 1, padding="same")(x)
    if bool(cfg.get("from_0_1", True)):
        x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)
    else:
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if drop > 0:
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenet_v2_transfer")

    if opt_name.lower() == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=lr)
    elif opt_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.get(opt_name)
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


register_model_builder("mobilenet_v2_transfer", build_mobilenet_v2_transfer)


# -------------------------
# Trainers
# -------------------------

def train_basic(
    model: tf.keras.Model, splits: DataSplits, cfg: Dict[str, Any]
) -> Tuple[tf.keras.Model, Any]:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    epochs: int = int(cfg.get("epochs", 5))
    batch_size: int = int(cfg.get("batch_size", 32))
    use_validation: bool = bool(cfg.get("use_validation", True))
    monitor: str = str(cfg.get("monitor", "val_loss"))
    patience: int = int(cfg.get("patience", 2))
    checkpoint_path: str = cfg.get("checkpoint_path", "best_model.keras")

    callbacks = []
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:  # Only create directory if path includes a directory
            _ensure_dir(checkpoint_dir)
        else:
            _ensure_dir(".")
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                mode="min" if "loss" in monitor else "max",
            )
        )
    if use_validation:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True))

    history = model.fit(
        splits.X_train,
        splits.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(splits.X_val, splits.y_val) if use_validation and splits.X_val.size else None,
        callbacks=callbacks,
        verbose=int(cfg.get("verbose", 1)),
    )
    return model, history


register_trainer("basic", train_basic)


def train_tfdata(
    model: tf.keras.Model, splits: DataSplits, cfg: Dict[str, Any]
) -> Tuple[tf.keras.Model, Any]:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    if bool(cfg.get("xla", False)):
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    epochs: int = int(cfg.get("epochs", 5))
    batch_size: int = int(cfg.get("batch_size", 32))
    use_validation: bool = bool(cfg.get("use_validation", True))
    monitor: str = str(cfg.get("monitor", "val_loss"))
    patience: int = int(cfg.get("patience", 2))
    checkpoint_path: str = cfg.get("checkpoint_path", "best_model.h5")
    buffer_size: int = int(cfg.get("shuffle_buffer", min(1000, int(splits.X_train.shape[0]) if splits.X_train.size else 1000)))
    cache = cfg.get("cache", True)
    prefetch: bool = bool(cfg.get("prefetch", True))

    aug_cfg: Dict[str, Any] = cfg.get("augment", {})
    flip_lr = bool(aug_cfg.get("flip_lr", True))
    flip_ud = bool(aug_cfg.get("flip_ud", False))
    rotate90 = bool(aug_cfg.get("rotate90", False))
    bright = float(aug_cfg.get("brightness", 0.0))
    contrast = float(aug_cfg.get("contrast", 0.0))

    def _map_fn(x, y):
        x = tf.cast(x, tf.float32)
        if flip_lr:
            x = tf.image.random_flip_left_right(x)
        if flip_ud:
            x = tf.image.random_flip_up_down(x)
        if rotate90:
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            x = tf.image.rot90(x, k)
        if bright > 0.0:
            x = tf.image.random_brightness(x, max_delta=bright)
        if contrast > 0.0:
            x = tf.image.random_contrast(x, lower=max(0.0, 1.0 - contrast), upper=1.0 + contrast)
        return x, y

    ds_train = tf.data.Dataset.from_tensor_slices((splits.X_train, splits.y_train))
    if cache:
        if isinstance(cache, str) and len(cache) > 0:
            ds_train = ds_train.cache(cache)
        else:
            ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size)
    if aug_cfg:
        ds_train = ds_train.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    if cfg.get("steps_per_epoch", None) is not None:
        ds_train = ds_train.repeat()
    if prefetch:
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = None
    if use_validation and splits.X_val.size:
        ds_val = tf.data.Dataset.from_tensor_slices((splits.X_val, splits.y_val))
        if cache:
            if isinstance(cache, str) and len(cache) > 0:
                ds_val = ds_val.cache(cache + "_val")
            else:
                ds_val = ds_val.cache()
        ds_val = ds_val.batch(batch_size, drop_remainder=False)
        if cfg.get("validation_steps", None) is not None:
            ds_val = ds_val.repeat()
        if prefetch:
            ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    callbacks = []
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            _ensure_dir(checkpoint_dir)
        else:
            _ensure_dir(".")
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                mode="min" if "loss" in monitor else "max",
            )
        )
    if use_validation:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True))

    history = model.fit(
        ds_train,
        epochs=epochs,
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        validation_data=ds_val if ds_val is not None else None,
        validation_steps=cfg.get("validation_steps", None),
        callbacks=callbacks,
        verbose=int(cfg.get("verbose", 1)),
    )
    return model, history


register_trainer("tfdata", train_tfdata)


# -------------------------
# Evaluators
# -------------------------

def evaluate_basic(
    model: tf.keras.Model, splits: DataSplits, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    results: Dict[str, Any] = {}
    if splits.X_test.size:
        loss, acc = model.evaluate(splits.X_test, splits.y_test, verbose=0)
        y_pred = model.predict(splits.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = splits.y_test
        cm = confusion_matrix(y_true, y_pred_classes)
        cr = classification_report(y_true, y_pred_classes, output_dict=False, zero_division=0)
        results.update(
            {
                "test_loss": float(loss),
                "test_accuracy": float(acc),
                "confusion_matrix": cm.tolist(),
                "classification_report": cr,
            }
        )
    elif splits.X_val.size:
        # No test set provided, use validation set
        loss, acc = model.evaluate(splits.X_val, splits.y_val, verbose=0)
        results.update({"val_loss": float(loss), "val_accuracy": float(acc)})
    else:
        # No test or validation set available
        raise ValueError("No test or validation data available for evaluation")
    return results


register_evaluator("basic", evaluate_basic)


# -------------------------
# Exporters
# -------------------------

def export_h5(model: tf.keras.Model, cfg: Dict[str, Any]) -> None:
    path: str = cfg.get("path", "best_model.keras")
    _ensure_dir(os.path.dirname(path) or ".")
    model.save(path)


def export_saved_model(model: tf.keras.Model, cfg: Dict[str, Any]) -> None:
    out_dir: str = cfg.get("out_dir", "saved_model")
    _ensure_dir(out_dir)
    # Try Keras 3 API first, fallback to TensorFlow 2.x
    if hasattr(model, 'export'):
        model.export(out_dir)
    else:
        # TensorFlow 2.x SavedModel format
        tf.saved_model.save(model, out_dir)


def export_tflite(model: tf.keras.Model, cfg: Dict[str, Any]) -> None:
    path: str = cfg.get("path", os.path.join("tflite", "model.tflite"))
    _ensure_dir(os.path.dirname(path) or ".")
    select_tf_ops: bool = bool(cfg.get("select_tf_ops", False))
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
    except Exception:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)


def export_tfjs(model: tf.keras.Model, cfg: Dict[str, Any]) -> None:
    out_dir: str = cfg.get("out_dir", "tfjs_model")
    _ensure_dir(out_dir)
    try:
        import tensorflowjs as tfjs

        # Save directly from Keras model
        tfjs.converters.save_keras_model(model, out_dir)
    except ImportError:
        # tensorflowjs not installed, save as H5 first then try CLI
        import subprocess
        h5_path = cfg.get('h5_path', 'best_model.keras')
        if not os.path.exists(h5_path):
            model.save(h5_path)
        try:
            cmd = ["tensorflowjs_converter", "--input_format=keras", h5_path, out_dir]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: TensorFlow.js export failed: {e}")
    except Exception as e:
        print(f"Warning: TensorFlow.js export failed: {e}")


register_exporter("save_h5", export_h5)
register_exporter("save_saved_model", export_saved_model)
register_exporter("save_tflite", export_tflite)
register_exporter("save_tfjs", export_tfjs)
