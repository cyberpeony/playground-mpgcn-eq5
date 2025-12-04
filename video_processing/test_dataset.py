from pathlib import Path
from torch.utils.data import DataLoader

from playground_dataset import PlaygroundPanoramicDataset


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]  
    train_csv = root / "data" / "train.csv"
    dataset = PlaygroundPanoramicDataset(csv_path=train_csv, V_max=32)
    print("Tamaño del dataset:", len(dataset))

    sample = dataset[0]
    print("sample_id:", sample["sample_id"])
    print("label:", sample["label"])
    print("J shape:", sample["J"].shape)
    print("B shape:", sample["B"].shape)
    print("JM shape:", sample["JM"].shape)
    print("BM shape:", sample["BM"].shape)
    print("A0 shape:", sample["A0"].shape)
    print("A_intra shape:", sample["A_intra"].shape)
    print("A_inter shape:", sample["A_inter"].shape)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(loader))
    print("\nBatch shapes:")
    print("J batch:", batch["J"].shape)   
    print("labels batch:", batch["label"].shape)
