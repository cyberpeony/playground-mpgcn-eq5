from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from playground_dataset import PlaygroundPanoramicDataset
from panoramic_graph import build_adjacency_matrices

def add_mpgcn_to_syspath():
    repo_root = Path(__file__).resolve().parents[2]  # .../equipo5
    mpgcn_src = repo_root / "MP-GCN" / "src"
    if str(mpgcn_src) not in sys.path:
        sys.path.append(str(mpgcn_src))
add_mpgcn_to_syspath()

from model.MPGCN.nets import MPGCN 

def main():
    repo_root = Path(__file__).resolve().parents[1]  # .../Socio-Formador-IA-Avanzada
    test_csv = repo_root / "data" / "test.csv"
    ckpt_path = repo_root / "checkpoints" / "mpgcn_playground_best.pth"

    print(f"Usando test.csv: {test_csv}")
    print(f"Usando checkpoint: {ckpt_path}")

    #params (los mismos que en train_mpgcn.py)
    T = 48
    V_max = 32 #17 joints + up to 15 objetos
    M = 4
    C = 2 #(x, y)
    I = 4 #streams: J, B, JM, BM

    #dataset y loader de test
    test_dataset = PlaygroundPanoramicDataset(csv_path=test_csv, V_max=V_max)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    #construir A global [3, V, V] (igual que en train_mpgcn.py)
    V_human = 17
    n_obj_max = V_max - V_human # 15
    A0, A_intra, A_inter = build_adjacency_matrices(V_human=V_human, n_obj=n_obj_max)
    A_np = np.stack([A0, A_intra, A_inter], axis=0) #[3, 32, 32]
    A = torch.from_numpy(A_np).float()

    # data_shape para MPGCN (num_input, num_channel, T, V, M)
    data_shape = (I, C, T, V_max, M)
    print("data_shape para MPGCN:", data_shape)
    print("A shape:", A.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    model = MPGCN(
        data_shape=data_shape,
        num_class=3,
        A=A.to(device),
        use_att=False,  
    ).to(device)

    # checkpoint 
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            J  = batch["J"].to(device) # [B, 2, T, V, M]
            B  = batch["B"].to(device)
            JM = batch["JM"].to(device)
            BM = batch["BM"].to(device)
            y  = batch["label"].to(device)
            #empaquetar (formato [N, I, C, T, V, M])
            x = torch.stack([J, B, JM, BM], dim=1)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    #para la matriz de confusión
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    np.save(repo_root / "data" / "preds_test.npy", all_preds)
    np.save(repo_root / "data" / "labels_test.npy", all_labels)
    print(f"Guardados preds_test.npy y labels_test.npy en {repo_root / 'data'}")


if __name__ == "__main__":
    main()
