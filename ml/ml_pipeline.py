from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf

from .ml_core import (
    DATA_SOURCES,
    AUGMENTERS,
    MODEL_BUILDERS,
    TRAINERS,
    EVALUATORS,
    EXPORTERS,
    DataSplits,
)


def _num_classes(y: np.ndarray) -> int:
    if y.size == 0:
        return 0
    return int(np.max(y) + 1)


def run(cfg: Dict[str, Any]) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    ds_name: str = data_cfg.get("source")
    if not ds_name or ds_name not in DATA_SOURCES:
        raise KeyError(f"Unknown data source: {ds_name}")
    splits: DataSplits = DATA_SOURCES[ds_name](data_cfg.get("params", {}))

    for aug in cfg.get("augmenters", []):
        name = aug.get("name")
        if not name:
            continue
        fn = AUGMENTERS.get(name)
        if not fn:
            raise KeyError(f"Unknown augmenter: {name}")
        splits = fn(splits, aug.get("params", {}))

    input_shape = tuple(splits.X_train.shape[1:])
    n_classes = _num_classes(splits.y_train)
    if n_classes <= 0:
        raise ValueError("Cannot infer number of classes from training labels")

    model_cfg: Dict[str, Any] = cfg.get("model", {})
    builder_name: str = model_cfg.get("builder")
    if not builder_name or builder_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model builder: {builder_name}")
    model = MODEL_BUILDERS[builder_name](input_shape, n_classes, model_cfg.get("params", {}))

    train_cfg: Dict[str, Any] = cfg.get("train", {})
    trainer_name: str = train_cfg.get("trainer")
    if not trainer_name or trainer_name not in TRAINERS:
        raise KeyError(f"Unknown trainer: {trainer_name}")
    model, history = TRAINERS[trainer_name](model, splits, train_cfg.get("params", {}))

    eval_cfg: Dict[str, Any] = cfg.get("evaluate", {})
    evaluator_name: str = eval_cfg.get("evaluator")
    if not evaluator_name or evaluator_name not in EVALUATORS:
        raise KeyError(f"Unknown evaluator: {evaluator_name}")
    results: Dict[str, Any] = EVALUATORS[evaluator_name](model, splits, eval_cfg.get("params", {}))

    for ex in cfg.get("export", []):
        name = ex.get("name")
        if not name:
            continue
        fn = EXPORTERS.get(name)
        if not fn:
            raise KeyError(f"Unknown exporter: {name}")
        fn(model, ex.get("params", {}))

    return model, results
