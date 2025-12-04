import numpy as np
import pandas as pd
from pathlib import Path

# build de los tensores de esqueletos desde CSVs (frame, person_id, x0..y16, y0..y16)
# tensor de forma [T, K_max, num_joints, 2]
# esto porque los x,y ya vienen normalizados desde el pipeline anterior
# reordena p/frame y person_id, elige K_max personas, muestrea a T frames, empaqueta en tensor
def csv_to_skeleton_tensor(csv_path, T=48, K_max=4, num_joints=17):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    expected_cols = ["frame", "person_id"] + \
        [f"x{i}" for i in range(num_joints)] + \
        [f"y{i}" for i in range(num_joints)]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path.name}: {missing}")
    #orden por tiempo/persona
    df = df.sort_values(["frame", "person_id"]).reset_index(drop=True)
    #frames únicos
    unique_frames = np.sort(df["frame"].unique())
    n_frames = len(unique_frames)
    if n_frames == 0:
        #csv vacío->todo ceros
        return np.zeros((T, K_max, num_joints, 2), dtype=np.float32)
    #personas más frecuentes hasta k max
    person_counts = df["person_id"].value_counts()
    selected_person_ids = list(person_counts.index[:K_max])
    person_id_to_idx = {pid: i for i, pid in enumerate(selected_person_ids)}

    skel = np.zeros((T, K_max, num_joints, 2), dtype=np.float32) #tensor de salida

    #temp mapping: 0..T-1 -> idx de frames reales
    frame_indices = np.linspace(0, n_frames - 1, num=T)
    frame_indices = np.round(frame_indices).astype(int)

    for t_model, idx in enumerate(frame_indices):
        frame_value = unique_frames[idx]
        df_f = df[df["frame"] == frame_value]
        for pid, m in person_id_to_idx.items():
            rows = df_f[df_f["person_id"] == pid]
            if rows.empty:
                continue  #esta persona no aparece en este frame
            row = rows.iloc[0]
            xs = np.array([row[f"x{j}"] for j in range(num_joints)], dtype=np.float32)
            ys = np.array([row[f"y{j}"] for j in range(num_joints)], dtype=np.float32)
            skel[t_model, m, :, 0] = xs
            skel[t_model, m, :, 1] = ys
    return skel

#recorre todos los csv y guarda un .npy [T,K_max,17,2] en out_dir con el mismo nombre base
def process_all_csv(
    csv_dir="skeletonCSV",
    out_dir="../data/npy",
    T=48,
    K_max=4,
    num_joints=17,
):
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Encontrados {len(csv_files)} csvs en {csv_dir.resolve()}")

    for csv_path in csv_files:
        print(f"\nProcesando {csv_path.name}")
        tensor = csv_to_skeleton_tensor(csv_path, T=T, K_max=K_max, num_joints=num_joints)
        out_path = out_dir / (csv_path.stem + ".npy")
        np.save(out_path, tensor)
        print(f"Guardado {out_path.resolve()}  shape={tensor.shape}")
    print("\n todos los csv ya son .npy")


if __name__ == "__main__":
    process_all_csv()