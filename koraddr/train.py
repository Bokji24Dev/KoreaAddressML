# -*- coding:utf-8 -*-
import os
import json
import torch
import pickle
import random
import pandas as pd
from tqdm import tqdm
from typing import List
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'


def load_data(filepaths: List[str]):
    rows = []
    # 피클 파일 로드
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            rows.extend(pickle.load(f))
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    # input_ids가 현재 List[int] 형태로 되어있는데 해당 형식을 각각 칼럼으로 만들고 Int64로 형 변환 후 데이터프레임으로 저장
    input_ids_df = pd.DataFrame(
        df["input_ids"].tolist(), columns=[f"input_ids_{i+1}" for i in range(32)]
    )
    # 기존의 DataFrame에 새롭게 만든 input_ids DataFrame을 합치고 기존에 List[int]로 된 input_ids 칼럼을 제거
    df = pd.concat([df, input_ids_df], axis=1).drop(columns=["input_ids"])
    # input_ids_1부터 input_ids_32 까지의 칼럼만 Feature로 남겨두기
    X = df.drop(columns=["text", "class"])
    # class 칼럼만 정답 데이터로 남겨두기
    y = df["class"]
    return X, y

seed = 42
random.seed(seed)
root_path = "dataset/archive_v3/"
files = [[] for _ in range(256)]
for location in os.listdir(root_path):
    for i in range(1, 256 + 1):
        files[i - 1].append(root_path + location + "/data_chunk_" + str(i) + ".pk")

num_classes = 280 # 17
batch_size = 256

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d((1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 1 * 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output(out)
        return out

# 모델 초기화
model = MLP(32, 64, 280)
model = SimpleCNN()

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
running_loss = 0.0

# 학습 루프
num_epochs = 20

PATH = "./weights/cnn/"

# files = files[:1]

result = []
accuracis = []

for i, file in tqdm(enumerate(files), total=len(files), position=0):
    X, y = load_data(file)
    X = X.dropna()
    y = y[X.index]
    
    # pandas DataFrame을 Tensor로 변환
    X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(2).unsqueeze(3)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in tqdm(range(num_epochs), position=1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 GPU로 이동
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        result.append({
            "round": i,
            "epoch": f"{epoch+1}/{num_epochs}",
            "loss": f"{epoch_loss:.4f}"
        })

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 GPU로 이동
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracis.append(f"{accuracy:.2f}")
    
    with open("accuracis.json", "w", encoding="utf-8") as f:
        json.dump(accuracis, f, ensure_ascii=False, indent=2)
    
    torch.save(model, PATH + f'model.{i}.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), PATH + f'model_state_dict.{i}.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, PATH + f'all.{i}.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
    