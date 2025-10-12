import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


DATA_SOURCES: Dict[str, Callable[[Dict[str, Any]], DataSplits]] = {}
AUGMENTERS: Dict[str, Callable[[DataSplits, Dict[str, Any]], DataSplits]] = {}
MODEL_BUILDERS: Dict[
    str,
    Callable[[Tuple[int, int, int], int, Dict[str, Any]], tf.keras.Model],
] = {}
TRAINERS: Dict[
    str,
    Callable[[tf.keras.Model, DataSplits, Dict[str, Any]], Tuple[tf.keras.Model, Any]],
] = {}
EVALUATORS: Dict[
    str, Callable[[tf.keras.Model, DataSplits, Dict[str, Any]], Dict[str, Any]]
] = {}
EXPORTERS: Dict[str, Callable[[tf.keras.Model, Dict[str, Any]], None]] = {}


def register_data_source(name: str, fn: Callable[[Dict[str, Any]], DataSplits]) -> None:
    DATA_SOURCES[name] = fn


def register_augmenter(
    name: str, fn: Callable[[DataSplits, Dict[str, Any]], DataSplits]
) -> None:
    AUGMENTERS[name] = fn


def register_model_builder(
    name: str,
    fn: Callable[[Tuple[int, int, int], int, Dict[str, Any]], tf.keras.Model],
) -> None:
    MODEL_BUILDERS[name] = fn


def register_trainer(
    name: str, fn: Callable[[tf.keras.Model, DataSplits, Dict[str, Any]], Tuple[tf.keras.Model, Any]]
) -> None:
    TRAINERS[name] = fn


def register_evaluator(
    name: str, fn: Callable[[tf.keras.Model, DataSplits, Dict[str, Any]], Dict[str, Any]]
) -> None:
    EVALUATORS[name] = fn


def register_exporter(name: str, fn: Callable[[tf.keras.Model, Dict[str, Any]], None]) -> None:
    EXPORTERS[name] = fn
