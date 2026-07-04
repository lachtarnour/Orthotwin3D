#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.datasets.teeth3ds_raw import FDI_LABELS
from src.utils.io import load_processed_sample
from src.utils.paths import get_processed_dir, get_split_dir


SPLITS = ("train", "val", "test")
POINT_KEYS = ("pos_raw", "pos", "normal")
LABEL_KEYS = ("y_binary", "y_fdi", "y_fdi_class", "y_instance", "source_indices")
REQUIRED_KEYS = {
    "scan_id", "patient_id", "jaw", "center", "scale", "fdi_to_class", "class_to_fdi",
    "landmarks_raw", "landmarks_norm", "landmark_to_tooth", "tooth_centers_raw", "tooth_centers_norm",
    *POINT_KEYS, *LABEL_KEYS,
}


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    split_dir = get_split_dir(args.split_source)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    patients: dict[str, set[str]] = {split: set() for split in SPLITS}
    total_files = total_landmarks = 0

    print("Checking processed dataset")
    print(f"processed_dir: {processed_dir}")
    print(f"split_dir:     {split_dir}\n")

    for split in SPLITS:
        files = sorted((processed_dir / split).glob("*.pt"))
        files = files[: args.limit] if args.limit else files
        if not files:
            all_errors.append(f"{split}: no .pt files found")
            continue

        split_landmarks = 0
        for path in files:
            errors, warnings, has_landmarks, patient_id = check_file(path)
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            split_landmarks += int(has_landmarks)
            if patient_id:
                patients[split].add(patient_id)

        total_files += len(files)
        total_landmarks += split_landmarks
        print(f"{split:5s}: {len(files):4d} files | {len(patients[split]):4d} patients | {split_landmarks:3d} with landmarks")

    all_errors.extend(patient_overlap_errors(patients))
    print_result(total_files, total_landmarks, all_warnings, all_errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple conformity check for processed OrthoTwin3D .pt files.")
    parser.add_argument("--processed_dir", default=str(get_processed_dir()))
    parser.add_argument("--split_source", default="patient_random")
    parser.add_argument("--limit", type=int, help="Check only the first N files per split.")
    return parser.parse_args()


def check_file(path: Path) -> tuple[list[str], list[str], bool, str | None]:
    try:
        sample = load_processed_sample(path)
    except Exception as exc:
        return [f"{path.name}: cannot load ({exc})"], [], False, None

    errors, warnings = check_sample(sample, path.name)
    return errors, warnings, bool(sample.get("landmarks_raw")), sample.get("patient_id")


def check_sample(sample: dict, name: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_KEYS - set(sample)
    if missing:
        return [f"{name}: missing keys {sorted(missing)}"], warnings

    n = check_shapes(sample, name, errors)
    if n is None:
        return errors, warnings

    check_finite(sample, name, errors)
    check_normals(sample["normal"], sample["y_fdi"], name, warnings)
    check_labels(sample, name, errors)
    check_metadata(sample, name, warnings, errors)
    check_curvature(sample, n, name, errors)
    return errors, warnings


def check_shapes(sample: dict, name: str, errors: list[str]) -> int | None:
    pos = sample["pos"]
    if tuple(pos.shape)[-1:] != (3,) or len(pos.shape) != 2:
        errors.append(f"{name}: pos must have shape [N, 3]")
        return None

    n = int(pos.shape[0])
    for key in POINT_KEYS:
        if tuple(sample[key].shape) != (n, 3):
            errors.append(f"{name}: {key} shape is {tuple(sample[key].shape)}, expected ({n}, 3)")
    for key in LABEL_KEYS:
        if tuple(sample[key].shape) != (n,):
            errors.append(f"{name}: {key} shape is {tuple(sample[key].shape)}, expected ({n},)")
    return n


def check_finite(sample: dict, name: str, errors: list[str]) -> None:
    for key in POINT_KEYS:
        if not torch.isfinite(sample[key]).all():
            errors.append(f"{name}: {key} contains NaN or inf")


def check_normals(normal: torch.Tensor, y_fdi: torch.Tensor, name: str, warnings: list[str]) -> None:
    zero_mask = torch.linalg.norm(normal.float(), dim=1) < 1e-6
    zero_on_tooth = int((zero_mask & (y_fdi > 0)).sum().item())
    zero_on_background = int((zero_mask & (y_fdi == 0)).sum().item())
    # Teeth3DS has a few degenerate background/gingiva vertices with no valid normal; they are not used as tooth geometry.
    if zero_on_tooth:
        warnings.append(f"{name}: {zero_on_tooth} zero normal vector(s) on tooth points")


def check_labels(sample: dict, name: str, errors: list[str]) -> None:
    y_fdi = sample["y_fdi"]
    y_class = sample["y_fdi_class"]
    valid_fdi = torch.tensor(FDI_LABELS, dtype=y_fdi.dtype)
    if not torch.isin(y_fdi, valid_fdi).all():
        errors.append(f"{name}: invalid FDI labels {sorted(set(y_fdi.tolist()) - set(FDI_LABELS))}")

    fdi_to_class = {int(k): int(v) for k, v in sample["fdi_to_class"].items()}
    class_to_fdi = {int(k): int(v) for k, v in sample["class_to_fdi"].items()}
    expected_classes = set(range(len(FDI_LABELS)))
    if set(fdi_to_class) != set(FDI_LABELS) or set(fdi_to_class.values()) != expected_classes:
        errors.append(f"{name}: invalid fdi_to_class")
    if set(class_to_fdi) != expected_classes or set(class_to_fdi.values()) != set(FDI_LABELS):
        errors.append(f"{name}: invalid class_to_fdi")

    expected_y_class = torch.empty_like(y_fdi)
    for fdi, class_id in fdi_to_class.items():
        expected_y_class[y_fdi == fdi] = class_id
    if not torch.equal(y_class, expected_y_class):
        errors.append(f"{name}: y_fdi_class is inconsistent with y_fdi")
    if not torch.equal(sample["y_binary"], (y_fdi > 0).long()):
        errors.append(f"{name}: y_binary is inconsistent with y_fdi")


def check_metadata(sample: dict, name: str, warnings: list[str], errors: list[str]) -> None:
    if sample["jaw"] not in {"upper", "lower"}:
        warnings.append(f"{name}: jaw is {sample['jaw']!r}")
    if not isinstance(sample["scale"], (float, int)) or float(sample["scale"]) <= 0:
        errors.append(f"{name}: scale must be positive")


def check_curvature(sample: dict, n: int, name: str, errors: list[str]) -> None:
    curvature = sample.get("curvature")
    if curvature is None:
        return
    if tuple(curvature.shape) != (n,):
        errors.append(f"{name}: curvature shape is {tuple(curvature.shape)}, expected ({n},)")
    elif not torch.isfinite(curvature).all():
        errors.append(f"{name}: curvature contains NaN or inf")


def patient_overlap_errors(patients: dict[str, set[str]]) -> list[str]:
    errors = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[left] & patients[right]
        if overlap:
            errors.append(f"patient overlap {left}/{right}: {sorted(overlap)[:10]}")
    return errors


def print_result(total_files: int, total_landmarks: int, warnings: list[str], errors: list[str]) -> None:
    print(f"\nTotal checked files: {total_files}")
    print(f"Files with landmarks: {total_landmarks}")
    print(f"Warnings: {len(warnings)}")
    print(f"Errors:   {len(errors)}")
    print_items("Warnings", warnings, limit=20)
    print_items("Errors", errors, limit=30)
    if errors:
        raise SystemExit(1)
    print("\nStatus: OK")


def print_items(title: str, items: list[str], limit: int) -> None:
    if not items:
        return
    print(f"\n{title}:")
    for item in items[:limit]:
        print(f"- {item}")
    if len(items) > limit:
        print(f"- ... {len(items) - limit} more")


if __name__ == "__main__":
    main()
