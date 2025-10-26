import os
import math
import re
from typing import Any, Dict, List, Tuple, Optional, Union
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


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    if y.size == 0:
        return {}
    classes, counts = np.unique(y, return_counts=True)
    total = float(y.size)
    num_classes = float(classes.size)
    return {
        int(cls): float(total / (num_classes * count))
        for cls, count in zip(classes, counts)
        if count > 0
    }


def _prepare_class_weight(value: Any, y: np.ndarray) -> Optional[Dict[int, float]]:
    if value is None or value is False:
        return None
    if value is True or value == "balanced" or value == "auto":
        weights = _compute_class_weights(y)
        return weights if weights else None
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {idx: float(w) for idx, w in enumerate(value)}
    return None


def _count_image_files(path: str, image_exts: Tuple[str, ...]) -> int:
    count = 0
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith(image_exts):
                count += 1
    return count


def _copy_image_tree(
    src_dir: str,
    dst_dir: str,
    image_exts: Tuple[str, ...],
    clear_dst: bool = False,
) -> int:
    if clear_dst and os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    _ensure_dir(dst_dir)

    copied = 0
    for root, _, files in os.walk(src_dir):
        for name in files:
            if not name.lower().endswith(image_exts):
                continue
            src_path = os.path.join(root, name)
            base, ext = os.path.splitext(name)
            dst_path = os.path.join(dst_dir, name)
            suffix = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{base}_{suffix}{ext}")
                suffix += 1
            shutil.copy2(src_path, dst_path)
            copied += 1

    if clear_dst and copied == 0:
        shutil.rmtree(dst_dir, ignore_errors=True)

    return copied


def _sanitize_label(name: str) -> str:
    sanitized = re.sub(r"[\s\-]+", "_", name.strip())
    sanitized = re.sub(r"[^0-9A-Za-z_]", "", sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _canonicalize_rice_label(name: str) -> str:
    sanitized = _sanitize_label(name)
    rice_prefix = "Rice___"
    sanitized_lower = sanitized.lower()
    if sanitized_lower.startswith(rice_prefix.lower()):
        body = sanitized[len(rice_prefix) :]
    else:
        body = sanitized
    parts = [part for part in body.split("_") if part]
    if parts:
        canonical_body = "_".join(part.capitalize() for part in parts)
    else:
        canonical_body = body
    return f"{rice_prefix}{canonical_body}" if canonical_body else sanitized


def _find_class_directory(root: str, class_name: str) -> str:
    sanitized_target = _sanitize_label(class_name).lower()
    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            if dirname == class_name:
                return os.path.join(dirpath, dirname)
        for dirname in dirnames:
            if _sanitize_label(dirname).lower() == sanitized_target:
                return os.path.join(dirpath, dirname)
    return ""


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


def _resolve_cache_target(cache_setting: Any, split: str, default_dir: str) -> Any:
    if isinstance(cache_setting, dict):
        target = cache_setting.get(split, cache_setting.get("default", True))
        return _resolve_cache_target(target, split, default_dir)

    if cache_setting in (None, False):
        return False
    if cache_setting is True:
        return True

    if isinstance(cache_setting, str):
        opt = cache_setting.strip()
        if not opt:
            return True
        lowered = opt.lower()
        if lowered in {"true", "memory", "ram"}:
            return True
        if lowered in {"false", "none", "off"}:
            return False
        if lowered in {"disk", "auto"}:
            cache_path = os.path.join(default_dir, f"{split}.cache")
            _ensure_dir(os.path.dirname(cache_path))
            return cache_path
        root, ext = os.path.splitext(opt)
        if ext:
            cache_path = opt
        else:
            cache_path = os.path.join(opt, f"{split}.cache")
        _ensure_dir(os.path.dirname(cache_path) or opt or ".")
        return cache_path

    return True


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
        - subset_fraction: Fraction of images per class (default: 1.0)
        - max_per_class: Cap images per class (0 disables, default: 0)
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


def data_kaggle_plant_disease_all(cfg: Dict[str, Any]) -> DataSplits:
    """Load Kaggle plant disease data filtered to potato, tomato, and rice leaf classes.

    Combines relevant classes from the PlantVillage dataset with the Rice Leaf
    Diseases dataset and prepares consistent train/val/test splits.

    Config params:
        - cache_dir: Directory to cache dataset (default: "data/plant_disease_all")
        - img_height: Image height (default: 256)
        - img_width: Image width (default: 256)
        - color_mode: "rgb" or "grayscale" (default: "rgb")
        - normalize: Normalize to [0,1] (default: True)
        - val_split: Validation split ratio (default: 0.15)
        - test_split: Test split ratio (default: 0.15)
        - random_state: Random seed (default: 42)
        - force_download: Force re-download even if cached (default: False)
        - subset_fraction: Fraction of images per class (default: 1.0)
        - max_images_per_class: Cap images per class (0 disables, default: 0)
    """
    from sklearn.model_selection import train_test_split

    cache_dir: str = cfg.get("cache_dir", "data/plant_disease_all")
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

    image_exts = (".png", ".jpg", ".jpeg", ".bmp")
    _ensure_dir(cache_dir)

    plant_classes = {
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato_Tomato_YellowLeaf_Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy",
    }

    allowed_prefixes = ("Potato___", "Tomato___", "Rice___")

    def _list_cached_classes(path: str) -> List[str]:
        if not os.path.isdir(path):
            return []
        classes: List[str] = []
        for entry in sorted(os.listdir(path)):
            cls_dir = os.path.join(path, entry)
            if not os.path.isdir(cls_dir):
                continue
            if _count_image_files(cls_dir, image_exts) > 0:
                classes.append(entry)
        return classes

    cached_class_names = _list_cached_classes(cache_dir)
    cached_set = set(cached_class_names)
    rice_present = any(name.startswith("Rice___") for name in cached_class_names)
    plant_subset_ok = plant_classes.issubset(cached_set)
    prefixes_ok = all(name.startswith(allowed_prefixes) for name in cached_class_names)
    data_ready = (
        len(cached_class_names) > 0
        and rice_present
        and plant_subset_ok
        and prefixes_ok
    )

    if not data_ready or force_download:
        print("Downloading Kaggle datasets...")
        try:
            import kagglehub

            plant_dataset_path = kagglehub.dataset_download(
                "karagwaanntreasure/plant-disease-detection"
            )
            print(f"Plant dataset downloaded to: {plant_dataset_path}")

            rice_dataset_path = kagglehub.dataset_download(
                "loki4514/rice-leaf-diseases-detection"
            )
            print(f"Rice dataset downloaded to: {rice_dataset_path}")

            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
            _ensure_dir(cache_dir)

            print("Extracting potato and tomato classes...")
            for cls in sorted(plant_classes):
                src_dir = _find_class_directory(plant_dataset_path, cls)
                if not src_dir:
                    print(f"  Warning: Class directory not found for {cls}")
                    continue
                dst_dir = os.path.join(cache_dir, cls)
                copied = _copy_image_tree(src_dir, dst_dir, image_exts, clear_dst=True)
                if copied == 0:
                    print(f"  Warning: No images copied for {cls}; removing directory")
                    shutil.rmtree(dst_dir, ignore_errors=True)
                else:
                    print(f"  {cls}: {copied} images")

            print("Extracting rice classes...")
            rice_class_sources: Dict[str, List[str]] = {}
            rice_prefix = "Rice___"

            for root, _, files in os.walk(rice_dataset_path):
                image_files = [f for f in files if f.lower().endswith(image_exts)]
                if not image_files:
                    continue
                class_token = _sanitize_label(os.path.basename(root))
                if not class_token:
                    continue
                class_name = _canonicalize_rice_label(class_token)
                rice_class_sources.setdefault(class_name, []).append(root)

            if not rice_class_sources:
                raise ValueError(
                    "Could not locate rice class directories with images in the downloaded dataset."
                )

            for class_name in sorted(rice_class_sources):
                dst_dir = os.path.join(cache_dir, class_name)
                first_source = True
                total_copied = 0
                for src_dir in rice_class_sources[class_name]:
                    copied = _copy_image_tree(
                        src_dir,
                        dst_dir,
                        image_exts,
                        clear_dst=first_source,
                    )
                    total_copied += copied
                    first_source = False
                if total_copied == 0:
                    print(
                        f"  Warning: No images copied for rice class {class_name}; removing directory"
                    )
                    shutil.rmtree(dst_dir, ignore_errors=True)
                else:
                    print(f"  {class_name}: {total_copied} images")

            cached_class_names = _list_cached_classes(cache_dir)
            cached_set = set(cached_class_names)
            rice_present = any(name.startswith("Rice___") for name in cached_class_names)
            if not cached_class_names or not rice_present or not plant_classes.issubset(cached_set):
                raise ValueError(
                    "Failed to prepare required classes. Please verify the downloaded datasets."
                )

        except ImportError:
            raise ImportError(
                "kagglehub is required to download the dataset. Install it with: pip install kagglehub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download/extract dataset: {e}")
    else:
        print(f"Using cached dataset from {cache_dir}")

    size = (img_width, img_height)
    X_all: List[np.ndarray] = []
    y_all: List[int] = []

    cached_classes = _list_cached_classes(cache_dir)
    if not cached_classes:
        raise ValueError("No classes with images were cached. Please verify the dataset contents.")

    class_groups: Dict[str, List[str]] = {}
    for cached_name in cached_classes:
        if cached_name.lower().startswith("rice___"):
            canonical = _canonicalize_rice_label(cached_name)
        else:
            canonical = cached_name
        class_groups.setdefault(canonical, []).append(cached_name)

    class_names = sorted(class_groups.keys())

    print(f"Loading images for {len(class_names)} classes...")

    for idx, canonical_name in enumerate(class_names):
        grouped_dirs = class_groups[canonical_name]
        file_paths: List[str] = []

        for cls in grouped_dirs:
            cls_dir = os.path.join(cache_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Skipping missing class directory: {cls_dir}")
                continue

            current_paths = [
                os.path.join(cls_dir, name)
                for name in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, name))
                and name.lower().endswith(image_exts)
            ]

            file_paths.extend(current_paths)

        if not file_paths:
            print(f"Warning: No images found for class {canonical_name}; skipping")
            continue

        if subset_fraction < 1.0:
            k = max(1, int(len(file_paths) * subset_fraction))
            rng.shuffle(file_paths)
            file_paths = file_paths[:k]

        if max_per_class > 0:
            file_paths = file_paths[:max_per_class]

        for path in file_paths:
            try:
                img = _read_image(path, size, color_mode)
                X_all.append(img)
                y_all.append(idx)
            except Exception as exc:
                print(f"Warning: Failed to read {path}: {exc}")

    if len(X_all) == 0:
        raise ValueError("No images loaded. Please check the dataset.")

    label_path = os.path.join(cache_dir, "label.txt")
    with open(label_path, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(class_names))

    X_all = np.stack(X_all, axis=0)
    y_all = np.asarray(y_all, dtype=np.int64)

    print(f"Loaded {len(X_all)} images across {len(class_names)} classes")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_split,
        random_state=random_state,
        stratify=y_all,
    )

    val_size_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp,
    )

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
register_data_source("kaggle_plant_disease_all", data_kaggle_plant_disease_all)


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
    from tensorflow.keras import (
        layers,
        models,
        optimizers,
        mixed_precision as mp,
        regularizers,
    )

    if bool(cfg.get("mixed_precision", False)):
        try:
            mp.set_global_policy("mixed_float16")
        except Exception:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[ml][mobilenet_v2_transfer] WARNING: No GPU detected; training will run on CPU.")
    else:
        gpu_names = ", ".join(dev.name for dev in gpus)
        print(f"[ml][mobilenet_v2_transfer] Using GPU(s): {gpu_names}")

    try:
        current_policy = mp.global_policy().name
    except Exception:
        current_policy = "unknown"
    if bool(cfg.get("mixed_precision", False)):
        if current_policy != "mixed_float16":
            print(
                f"[ml][mobilenet_v2_transfer] WARNING: Mixed precision requested but active policy is {current_policy}."
            )
        else:
            print("[ml][mobilenet_v2_transfer] Mixed precision policy active: mixed_float16")
    else:
        print(f"[ml][mobilenet_v2_transfer] Mixed precision policy active: {current_policy}")

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
    dense_l2 = float(cfg.get("dense_l2", 0.0))
    classifier_l2 = float(cfg.get("classifier_l2", dense_l2))

    dense_regularizer = regularizers.l2(dense_l2) if dense_l2 > 0 else None
    classifier_regularizer = (
        regularizers.l2(classifier_l2) if classifier_l2 > 0 else None
    )

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
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=dense_regularizer,
    )(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=classifier_regularizer,
    )(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenet_v2_transfer")

    if opt_name.lower() == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=lr)
    elif opt_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.get(opt_name)
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss_obj, metrics=["accuracy"])
    return model


register_model_builder("mobilenet_v2_transfer", build_mobilenet_v2_transfer)


# -------------------------
# Trainers
# -------------------------

def train_basic(
    model: tf.keras.Model, splits: DataSplits, cfg: Dict[str, Any]
) -> Tuple[tf.keras.Model, Any]:
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )

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

    reduce_lr_cfg = cfg.get("reduce_lr_on_plateau", None)
    if isinstance(reduce_lr_cfg, bool):
        reduce_lr_cfg = {} if reduce_lr_cfg else None
    if reduce_lr_cfg is not None:
        reduce_lr_params = {
            "monitor": monitor,
            "factor": 0.3,
            "patience": 3,
            "min_lr": 3e-6,
            "verbose": int(cfg.get("verbose", 1)) > 0,
        }
        if isinstance(reduce_lr_cfg, dict):
            reduce_lr_params.update(reduce_lr_cfg)
            reduce_lr_params.setdefault("monitor", monitor)
        callbacks.append(ReduceLROnPlateau(**reduce_lr_params))

    class_weight_cfg = cfg.get("class_weight")
    class_weight = _prepare_class_weight(class_weight_cfg, splits.y_train)

    history = model.fit(
        splits.X_train,
        splits.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(splits.X_val, splits.y_val) if use_validation and splits.X_val.size else None,
        callbacks=callbacks,
        verbose=int(cfg.get("verbose", 1)),
        class_weight=class_weight,
    )
    return model, history


register_trainer("basic", train_basic)


def train_tfdata(
    model: tf.keras.Model, splits: DataSplits, cfg: Dict[str, Any]
) -> Tuple[tf.keras.Model, Any]:
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )

    if bool(cfg.get("xla", False)):
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    try:
        if tf.config.optimizer.get_jit():
            print("[ml][train_tfdata] XLA JIT compilation enabled.")
        elif bool(cfg.get("xla", False)):
            print("[ml][train_tfdata] WARNING: XLA requested but not enabled.")
    except Exception:
        if bool(cfg.get("xla", False)):
            print("[ml][train_tfdata] WARNING: Unable to query XLA status.")

    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        print("[ml][train_tfdata] WARNING: No GPU detected; training will run on CPU.")
    else:
        gpu_names = ", ".join(dev.name for dev in gpu_devices)
        print(f"[ml][train_tfdata] Using GPU(s): {gpu_names}")

    try:
        policy_name = tf.keras.mixed_precision.global_policy().name
    except Exception:
        policy_name = "unknown"
    print(f"[ml][train_tfdata] Mixed precision policy: {policy_name}")

    epochs: int = int(cfg.get("epochs", 5))
    batch_size: int = int(cfg.get("batch_size", 32))
    use_validation: bool = bool(cfg.get("use_validation", True))
    monitor: str = str(cfg.get("monitor", "val_loss"))
    patience: int = int(cfg.get("patience", 2))
    checkpoint_path: str = cfg.get("checkpoint_path", "best_model.h5")
    buffer_size: int = int(cfg.get("shuffle_buffer", min(1000, int(splits.X_train.shape[0]) if splits.X_train.size else 1000)))
    cache = cfg.get("cache", True)
    cache_val_override = cfg.get("cache_val", cache)
    cache_dir = cfg.get("cache_dir", os.path.join(".tf_cache", model.name or "model_cache"))
    _ensure_dir(cache_dir)

    prefetch: bool = bool(cfg.get("prefetch", True))

    def _apply_cache(ds: tf.data.Dataset, target: Any) -> tf.data.Dataset:
        if target is True:
            return ds.cache()
        if isinstance(target, str) and target:
            return ds.cache(target)
        return ds

    ds_train = tf.data.Dataset.from_tensor_slices((splits.X_train, splits.y_train))
    train_cache_target = _resolve_cache_target(cache, "train", cache_dir)
    ds_train = _apply_cache(ds_train, train_cache_target)
    ds_train = ds_train.shuffle(buffer_size)

    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    if cfg.get("steps_per_epoch", None) is not None:
        ds_train = ds_train.repeat()
    if prefetch:
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = None
    if use_validation and splits.X_val.size:
        ds_val = tf.data.Dataset.from_tensor_slices((splits.X_val, splits.y_val))
        val_cache_target = _resolve_cache_target(cache_val_override, "val", cache_dir)
        if (
            isinstance(train_cache_target, str)
            and isinstance(val_cache_target, str)
            and train_cache_target == val_cache_target
        ):
            val_cache_target = f"{val_cache_target}_val"
            _ensure_dir(os.path.dirname(val_cache_target) or ".")
        ds_val = _apply_cache(ds_val, val_cache_target)
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

    reduce_lr_cfg = cfg.get("reduce_lr_on_plateau", None)
    if isinstance(reduce_lr_cfg, bool):
        reduce_lr_cfg = {} if reduce_lr_cfg else None
    if reduce_lr_cfg is not None:
        reduce_lr_params = {
            "monitor": monitor,
            "factor": 0.3,
            "patience": 3,
            "min_lr": 3e-6,
            "verbose": int(cfg.get("verbose", 1)) > 0,
        }
        if isinstance(reduce_lr_cfg, dict):
            reduce_lr_params.update(reduce_lr_cfg)
            reduce_lr_params.setdefault("monitor", monitor)
        callbacks.append(ReduceLROnPlateau(**reduce_lr_params))

    class_weight_cfg = cfg.get("class_weight")
    class_weight = _prepare_class_weight(class_weight_cfg, splits.y_train)

    history = model.fit(
        ds_train,
        epochs=epochs,
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        validation_data=ds_val if ds_val is not None else None,
        validation_steps=cfg.get("validation_steps", None),
        callbacks=callbacks,
        verbose=int(cfg.get("verbose", 1)),
        class_weight=class_weight,
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
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        if select_tf_ops:
            raise RuntimeError("TFLite conversion failed even with SELECT_TF_OPS enabled.") from exc
        raise RuntimeError(
            "TFLite conversion failed when restricted to built-in ops. "
            "Review model layers for TensorFlow Lite compatibility or enable 'select_tf_ops'."
        ) from exc
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
