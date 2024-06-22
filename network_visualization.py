import torch
import torch.nn as nn
from torchviz import make_dot

num_classes = 17
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


model = SimpleCNN()


# 크기가 32인 정수 리스트 생성
batch_size = 1
channels = 32
height = 28
width = 28

x = torch.zeros((batch_size, channels, height, width))
y = model(x)


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


dot = make_dot(
    y, params=dict(list(model.named_parameters())), show_attrs=False, show_saved=False
)
resize_graph(dot, size_per_element=100, min_size=10)
dot.render("rnn_torchviz", format="png")
