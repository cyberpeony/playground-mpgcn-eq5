from pathlib import Path
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

from playground_dataset import PlaygroundPanoramicDataset
from panoramic_graph import build_adjacency_matrices


#src de MP-GCN 
def add_mpgcn_to_syspath():
    repo_root = Path(__file__).resolve().parents[2] #.../equipo5
    mpgcn_src = repo_root / "MP-GCN" / "src"
    if str(mpgcn_src) not in sys.path:
        sys.path.append(str(mpgcn_src))


add_mpgcn_to_syspath()

from model.MPGCN.nets import MPGCN 


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        J  = batch["J"].to(device) #[B, 2, T, V, M]
        B  = batch["B"].to(device)
        JM = batch["JM"].to(device)
        BM = batch["BM"].to(device)
        y  = batch["label"].to(device)

        #empaquetar en [N, I, C, T, V, M]
        x = torch.stack([J, B, JM, BM], dim=1)

        optimizer.zero_grad()
        logits, _ = model(x) #MPGCN devuelve (x,feature)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            J  = batch["J"].to(device)
            B  = batch["B"].to(device)
            JM = batch["JM"].to(device)
            BM = batch["BM"].to(device)
            y  = batch["label"].to(device)

            x = torch.stack([J, B, JM, BM], dim=1)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed = 7
    print("Seed:", seed)
    set_seed(seed)
    #rutas base
    repo_root = Path(__file__).resolve().parents[1] #.../Socio-Formador-IA-Avanzada
    train_csv = repo_root / "data" / "train.csv"
    val_csv   = repo_root / "data" / "val.csv"

    #params
    T = 48
    V_max = 32 #17 joints + hasta 15 obj
    M = 4
    C = 2 #(x, y)
    I = 4 #streams: J, B, JM, BM

    #ds y loaders
    train_dataset = PlaygroundPanoramicDataset(csv_path=train_csv, V_max=V_max)
    val_dataset   = PlaygroundPanoramicDataset(csv_path=val_csv,   V_max=V_max)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)

    #definir data_shape para MPGCN (num_input, num_channel, T, V, M)
    data_shape = (I, C, T, V_max, M)
    print("data_shape para MPGCN:", data_shape)

    #construir A global [3, V, V]
    V_human = 17
    n_obj_max = V_max - V_human #15
    A0, A_intra, A_inter = build_adjacency_matrices(V_human=V_human, n_obj=n_obj_max)
    A_np = np.stack([A0, A_intra, A_inter], axis=0) #[3, 32, 32]
    A = torch.from_numpy(A_np).float()
    print("A shape:", A.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    model = MPGCN(
        data_shape=data_shape,
        num_class=3,
        A=A,
        use_att=False, #sin atención para que no pida parts y reduct_ratio
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 40 

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )
        dt = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "data_shape": data_shape,
                "num_classes": 3,
                "epoch": epoch,
                "val_acc": val_acc,
            }

        print(
            f"[Época {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
            f"best_val_acc={best_val_acc:.3f} | "
            f"tiempo={dt:.1f}s"
        )

    #solo el mejor modelo según val_acc
    ckpt_path = repo_root / "checkpoints" / "mpgcn_playground_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    print(f" Mejor checkpoint MPGCN guardado en {ckpt_path} (val_acc={best_val_acc:.3f})")