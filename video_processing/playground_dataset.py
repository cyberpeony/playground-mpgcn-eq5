from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from panoramic_graph import build_panoramic_graph_from_pano

#mapeo de etiquetas string a idx entero
LABEL_MAP = {
    "Transit": 0,
    "Social_People": 1,
    "Play_Object_Normal": 2,
}

#ds para las escenas, lee data/train.csv o data/val.csv, carga el .npy panorámico, construye:
# J, B, JM, BM   (cada uno [C, T, V', M])
# A0, A_intra, A_inter ([V', V'])
#y devuelve todo junto con la etiqueta
class PlaygroundPanoramicDataset(Dataset):
    def __init__(
        self,
        csv_path, #ruta a train.csv o val.csv
        data_root=None,
        V_human=17, #joints
        V_max=22,
        transform=None, #placeholder for now
    ):
        self.csv_path = Path(csv_path).resolve()
        if data_root is None:
            self.data_root = self.csv_path.parent
        else:
            self.data_root = Path(data_root).resolve()
        self.V_human = V_human
        self.V_max = V_max
        self.transform = transform
        self.samples = []
        with self.csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_str = row["label"]
                if label_str not in LABEL_MAP:
                    raise ValueError(f"tag desconocida {label_str}")
                label_idx = LABEL_MAP[label_str]
                npy_rel = row["path"]
                npy_path = self.data_root / npy_rel
                self.samples.append(
                    {
                        "sample_id": row["sample_id"],
                        "npy_path": npy_path,
                        "label_str": label_str,
                        "label_idx": label_idx,
                        "camera": row.get("camera", "unknown"),
                    }
                )

        print(
            f"dataset {self.csv_path} c/ {len(self.samples)} muestras."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npy_path = sample["npy_path"]
        label_idx = sample["label_idx"]
        pano = np.load(npy_path) #tensor panoram.
        #streams y adj
        J, B, JM, BM, A0, A_intra, A_inter = build_panoramic_graph_from_pano(
            pano, V_human=self.V_human
        )
        #a tensores de pytorch
        J_t = torch.from_numpy(J).float()
        B_t = torch.from_numpy(B).float()
        JM_t = torch.from_numpy(JM).float()
        BM_t = torch.from_numpy(BM).float()
        A0_t = torch.from_numpy(A0).float()
        A_intra_t = torch.from_numpy(A_intra).float()
        A_inter_t = torch.from_numpy(A_inter).float()
        y = torch.tensor(label_idx, dtype=torch.long)
        #padding en la dimensión V' para unir a V_max
        V_prime = J_t.shape[2] #[C, T, V', M]
        V_max = self.V_max

        if V_prime < V_max:
            pad = V_max - V_prime

            #streams [C, T, V', M] -> [C, T, V_max, M]
            def pad_stream(X):
                C, T, V, M = X.shape
                out = torch.zeros((C, T, V_max, M), dtype=X.dtype)
                out[:, :, :V, :] = X
                return out
            J_t = pad_stream(J_t)
            B_t = pad_stream(B_t)
            JM_t = pad_stream(JM_t)
            BM_t = pad_stream(BM_t)
            #adj matrix [V', V'] -> [V_max, V_max]
            def pad_adj(A, self_loop=False):
                V, _ = A.shape
                out = torch.zeros((V_max, V_max), dtype=A.dtype)
                out[:V, :V] = A
                if self_loop:
                    for i in range(V, V_max):
                        out[i, i] = 1.0
                return out
            A0_t = pad_adj(A0_t, self_loop=True)  
            A_intra_t = pad_adj(A_intra_t, self_loop=False)
            A_inter_t = pad_adj(A_inter_t, self_loop=False)

        sample_out = {
            "J": J_t,             
            "B": B_t,
            "JM": JM_t,
            "BM": BM_t,
            "A0": A0_t,          
            "A_intra": A_intra_t,
            "A_inter": A_inter_t,
            "label": y,
            "sample_id": sample["sample_id"],
            "camera": sample["camera"],
            "npy_path": str(npy_path),
        }

        if self.transform is not None:
            sample_out = self.transform(sample_out)
        return sample_out
