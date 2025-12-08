from pathlib import Path
import csv
import re
import random
from collections import defaultdict

#sacar clase del nombre del archivo
def infer_label_from_filename(name_stem: str) -> str:
    parts = name_stem.split("_")
    idx_token = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            idx_token = i
            break

    if idx_token is None or idx_token == len(parts) - 1:
        raise ValueError(f"No pude inferir etiqueta de: {name_stem}")

    label_parts = parts[idx_token + 1 :]
    label = "_".join(label_parts)
    return label

#sacar camara del stem
def infer_camera_from_filename(name_stem: str) -> str:
    m = re.search(r"cam(\d+)", name_stem)
    if not m:
        return "unknown"
    cam_num = m.group(1)
    return f"columpioscam{cam_num}"

#recorriendo los .npy en pano_dir e infiriendo etiqueta y camara
#y build de train.csv, val.csv y test.csv con split estratificado por clase
def build_train_val_test(
    pano_dir: Path,
    out_train_csv: Path,
    out_val_csv: Path,
    out_test_csv: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio debe ser < 1"
    random.seed(seed)
    pano_files = sorted(pano_dir.glob("*.npy"))
    print(f"Encontrados {len(pano_files)} .npy en {pano_dir}")

    #por etiqueta
    by_label = defaultdict(list)
    for npy_path in pano_files:
        stem = npy_path.stem  #sin .npy
        label = infer_label_from_filename(stem)
        camera = infer_camera_from_filename(stem)
        sample = {
            "sample_id": stem,
            "path": str(npy_path.relative_to(pano_dir.parent)),  #relativo a data
            "label": label,
            "camera": camera,
        }
        by_label[label].append(sample)
    #split por clase
    train_rows = []
    val_rows = []
    test_rows = []
    print("\nRec. por clase y split:")

    for label, samples in by_label.items():
        n = len(samples)
        random.shuffle(samples)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        #1 ejemp min. en cada split
        if n_train == 0 and n > 0:
            n_train = 1
        if n_val == 0 and n - n_train > 1:
            n_val = 1
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 #al menos 1 para test 

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]
        train_rows.extend(train_samples)
        val_rows.extend(val_samples)
        test_rows.extend(test_samples)
        print(
            f"  {label}: total={n}, "
            f"train={len(train_samples)}, "
            f"val={len(val_samples)}, "
            f"test={len(test_samples)}"
        )

    header = ["sample_id", "path", "label", "camera"] #csvs
    out_train_csv.parent.mkdir(parents=True, exist_ok=True)
    out_val_csv.parent.mkdir(parents=True, exist_ok=True)
    out_test_csv.parent.mkdir(parents=True, exist_ok=True)

    def save_csv(rows, out_path):
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Guardado {out_path.name} en: {out_path}")
    save_csv(train_rows, out_train_csv)
    save_csv(val_rows, out_val_csv)
    save_csv(test_rows, out_test_csv)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    pano_dir = root / "data" / "panoramic_npy"

    out_train_csv = root / "data" / "train.csv"
    out_val_csv = root / "data" / "val.csv"
    out_test_csv = root / "data" / "test.csv"
    build_train_val_test(
        pano_dir,
        out_train_csv,
        out_val_csv,
        out_test_csv,
        train_ratio=0.7,
        val_ratio=0.15,
    )