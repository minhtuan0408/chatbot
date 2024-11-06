import torch
from torch.utils.data import Dataset, DataLoader

# Ví dụ tạo một dataset tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


# Ví dụ dữ liệu và nhãn
data = torch.randn(1, 3, 32, 32)  # 100 ảnh RGB có kích thước 32x32
labels = torch.randint(0, 10, (1,))  # 100 nhãn ngẫu nhiên trong khoảng 0-9

# Khởi tạo dataset
dataset = CustomDataset(data, labels)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print("Inputs:", inputs.size())  # Kích thước của batch dữ liệu
    print("Labels:", labels.size())  # Kích thước của batch nhãn
    print("Inputs Data:", inputs)  # In ra dữ liệu của batch
    print("Labels Data:", labels)  # In ra nhãn của batch
    print("-" * 50)