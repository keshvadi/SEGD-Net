"""BreaKHis/BACH image discovery, loading, and group-aware stratified folds."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
BACH_LABELS = {
    "normal": 0,
    "benign": 0,
    "in situ": 1,
    "insitu": 1,
    "invasive": 1,
    "malignant": 1,
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    group: str


def _breakhis_label(path: Path) -> int:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if any(part in {"benign", "b"} for part in parts) or re.search(
        r"(^|[_-])b([_-]|$)", name
    ):
        return 0
    if any(part in {"malignant", "m"} for part in parts) or re.search(
        r"(^|[_-])m([_-]|$)", name
    ):
        return 1
    raise ValueError(f"Cannot infer BreaKHis class from path: {path}")


def _bach_label(path: Path) -> int:
    for part in reversed(path.parts[:-1]):
        normalized = re.sub(r"[_-]+", " ", part.lower()).strip()
        if normalized in BACH_LABELS:
            return BACH_LABELS[normalized]
    stem = re.sub(r"[_-]+", " ", path.stem.lower())
    for name, label in BACH_LABELS.items():
        if name in stem:
            return label
    raise ValueError(f"Cannot infer BACH class from path: {path}")


def _breakhis_group(path: Path) -> str:
    match = re.search(r"(SOB_[A-Z]_[A-Z0-9]+-\d+-[A-Z0-9]+)", path.name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    for part in path.parts:
        match = re.search(r"(patient[_-]?\d+)", part, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    raise ValueError(f"Cannot infer patient group from BreaKHis path: {path}")


def discover_images(root: str | Path, dataset: str) -> list[ImageRecord]:
    """Scan images without downloading data; fail on unknown labels or empty roots."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_path}")
    dataset_name = dataset.lower()
    if dataset_name not in {"breakhis", "bach"}:
        raise ValueError("dataset must be either 'breakhis' or 'bach'")

    records: list[ImageRecord] = []
    for path in sorted(
        p for p in root_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    ):
        if dataset_name == "breakhis":
            label, group = _breakhis_label(path), _breakhis_group(path)
        else:
            label = _bach_label(path)
            group = path.stem
        records.append(ImageRecord(path=path, label=label, group=group))
    if not records:
        raise FileNotFoundError(f"No supported images found under {root_path}")
    return records


class HistologyImageDataset(Dataset):
    """PIL-backed dataset that returns image, label, path, and group metadata."""

    def __init__(
        self,
        records: list[ImageRecord],
        transform: Callable | None = None,
        indices: list[int] | np.ndarray | None = None,
        graph_transform: Callable | None = None,
        label_overrides: dict[int, int] | None = None,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        self.records = records
        self.indices = (
            np.arange(len(records)) if indices is None else np.asarray(indices)
        )
        self.transform = transform
        self.graph_transform = graph_transform
        self.label_overrides = label_overrides or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        record = self.records[int(self.indices[index])]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        graph = (
            self.graph_transform(image.copy(), record) if self.graph_transform else None
        )
        if self.transform is not None:
            image = self.transform(image)
        sample = {
            "image": image,
            "label": self.label_overrides.get(int(self.indices[index]), record.label),
            "path": str(record.path),
            "group": record.group,
        }
        if graph is not None:
            sample["graph"] = graph
        return sample


def build_folds(
    records: list[ImageRecord], n_splits: int = 5, seed: int = 17
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified group folds, keeping all images from each patient together."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.group for record in records])
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"{n_splits} folds require at least {n_splits} distinct groups; "
            f"found {len(np.unique(groups))}"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train, test)
        for train, test in splitter.split(np.zeros(len(labels)), labels, groups)
    ]


def build_train_validation_test_split(
    records: list[ImageRecord], seed: int = 17
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create group-stratified 70/10/20 train/validation/test indices."""
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.group for record in records])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val, test = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner_labels = labels[train_val]
    inner_groups = groups[train_val]
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
    train_local, validation_local = next(
        inner.split(np.zeros(len(train_val)), inner_labels, inner_groups)
    )
    train = train_val[train_local]
    validation = train_val[validation_local]
    return train, validation, test


def build_cross_validation_splits(
    records: list[ImageRecord], n_splits: int = 5, seed: int = 17
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return repeated group-stratified 70/10/20 train/validation/test partitions."""
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.group for record in records])
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (train_val, test) in enumerate(
        outer.split(np.zeros(len(labels)), labels, groups)
    ):
        train_labels = labels[train_val]
        train_groups = groups[train_val]
        if len(np.unique(train_groups)) < 8:
            raise ValueError(
                "At least eight training groups are needed to create a grouped "
                "approximately 10% validation split"
            )
        inner = StratifiedGroupKFold(
            n_splits=8, shuffle=True, random_state=seed + fold + 1
        )
        train_local, validation_local = next(
            inner.split(np.zeros(len(train_val)), train_labels, train_groups)
        )
        splits.append((train_val[train_local], train_val[validation_local], test))
    return splits


def collate_histology(samples: list[dict]) -> dict:
    """Stack image tensors while retaining per-patch variable-size graph data."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "label": torch.tensor(
            [sample["label"] for sample in samples], dtype=torch.long
        ),
        "path": [sample["path"] for sample in samples],
        "group": [sample["group"] for sample in samples],
    }
    if all("graph" in sample for sample in samples):
        batch["graphs"] = [
            (
                torch.as_tensor(sample["graph"][0], dtype=torch.float32),
                torch.as_tensor(sample["graph"][1], dtype=torch.float32),
            )
            for sample in samples
        ]
    elif any("graph" in sample for sample in samples):
        raise ValueError("a batch cannot mix samples with and without graph data")
    return batch
