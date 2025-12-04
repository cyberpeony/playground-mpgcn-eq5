from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from playground_dataset import PlaygroundPanoramicDataset

#baseline, usa solo el stream J (joints) 
#aplana [B, 2, T, V, M] a [B, D]
#pasa por un MLP -> 3 clases
class SimpleBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, J):
        #J:[B, 2, T, V, M]
        B = J.shape[0]
        x = J.view(B, -1)#[B, in_dim]
        out = self.net(x)
        return out

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        J = batch["J"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(J)              
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
            J = batch["J"].to(device)   
            y = batch["label"].to(device)

            logits = model(J)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]  
    train_csv = root / "data" / "train.csv"
    val_csv = root / "data" / "val.csv"
    #ds con V_max = 32 (17 joints + up to 15 obj)
    train_dataset = PlaygroundPanoramicDataset(csv_path=train_csv, V_max=32)
    val_dataset = PlaygroundPanoramicDataset(csv_path=val_csv, V_max=32)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    #shape de 1 batch para calcular in_dim
    sample_batch = next(iter(train_loader))
    J_sample = sample_batch["J"] 
    _, C, T, V, M = J_sample.shape
    in_dim = C * T * V * M
    print(f"in_dim para el baseline: {in_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    model = SimpleBaseline(in_dim=in_dim, hidden_dim=512, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 5 

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(
            f"[Época {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
            f"tiempo={dt:.1f}s"
        )
    #checkpoint sencillo
    ckpt_path = root / "checkpoints" / "baseline_simple.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "num_classes": 3,
        },
        ckpt_path,
    )
    print(f"checkpoint en {ckpt_path}")
