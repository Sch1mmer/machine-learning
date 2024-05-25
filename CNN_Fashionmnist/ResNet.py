import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), op=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        ) if op else None
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        if self.conv0:
            x = self.conv0(x)
        return self.ReLU(x + x1)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
        )
        self.ResNetBlock1 = nn.Sequential(
            ResNetBlock(64, 64, stride=(1, 1), op=False),
            ResNetBlock(64, 64, stride=(1, 1), op=False)
        )
        self.ResNetBlock2 = nn.Sequential(
            ResNetBlock(64, 128, stride=(2, 2), op=True), # 上一层的输出通道必须等于下一层的输入通道
            ResNetBlock(128, 128, stride=(1, 1), op=False),
            ResNetBlock(128, 256, stride=(2, 2), op=True),
            ResNetBlock(256, 256, stride=(1, 1), op=False),
            ResNetBlock(256,512, stride=(2, 2), op=True),
            ResNetBlock(512, 512, stride=(1, 1), op=False),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)  # 1 --> 64,
        x = self.ResNetBlock1(x)  # 64 --> 64， 64 --> 64,
        x = self.ResNetBlock2(x) # 64 --> 128， 128 --> 128 ， 128 --> 256 ， 256 --> 256 ， 256 --> 512 ， 512 --> 512
        x = self.fc(x)  # 512 -- > 10,实验组ResNet_6通道数减半
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)

net = ResNet().to(device)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.2).to(device)


# 实验组ResNet_3：SGD, 对照组ResNet：AdamW
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3) # 学习率，对照组ResNet： 0.001，实验组ResNet_1： 0.01,ResNet_2： 0.0001
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # 衰减权重
my_writer = SummaryWriter("tf-logs")

epoch = 20 # 训练轮数
batch_size = 32 # 同一批次训练数据的数量 实验组ResNet_4：64, 实验组ResNet_5：16, 对照组ResNet：32

train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
print('ResNet')

for step in range(epoch):
    # 训练
    net.train()
    train_loss, train_acc = 0, 0
    for img, label in train_dataloader:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        y = net.forward(img)
        loss = loss_function(y, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()  # 计算准确率
    scheduler.step()
    # print(f"Train Epoch: {step}, Loss: {train_loss / len(train_dataloader)}, Acc: {train_acc / len(mnist_train)}")
    # 测试
    net.eval()
    test_loss, test_acc = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        y = net.forward(img)
        loss = loss_function(y, label)
        test_loss += loss.item()
        test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()
    print(f"周期: {step}, 误差: {test_loss / len(test_dataloader)}, 精度: {test_acc / len(mnist_test)}")

    # 统计
    my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
    my_writer.add_scalars("Acc", {"train": train_acc / len(mnist_train), "test": test_acc / len(mnist_test)}, step)

# 保存模型
torch.save(net.state_dict(), "save/ResNet.pt")
