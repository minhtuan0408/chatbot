import json
import numpy as np
import random

#Tiền xử lý và chuẩn bị dữ liệu: 
#tokenization và vector hóa.
from mynltk import tokenize, bag_of_words

#thư viện huấn luyên mô hình học sâu (deep learning)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Model của tôi
from myModel import NeuralNet

with open ('script.json', 'r',encoding='utf-8') as f:
    script = json.load(f) 


all_words =[]
tags =[]
xy =[]

#Đọc file lấy tag và patterns trong file json
#x = word, y = tag
for intent in script['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        #tách từ trong pattern
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

#Bỏ các kí tự
ignore_words = ['?', '!', '.', ',']


#Tranning data
#x = word, y = tag
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    #trả về danh sách các vector
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


#
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.n_samples

#Hyperparameter
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001

#Số lần lặp lại dữ liệu
num_epochs =1000

print(input_size, len(all_words))
print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)

#kiểm tra mô hình huấn luyên bằng GPU hoặc cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Triển khai myModel
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Triển khai hàm mất mát
criterion = nn.CrossEntropyLoss()

# Tối ưu hóa
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward và tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
