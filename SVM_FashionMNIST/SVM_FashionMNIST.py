import numpy as np
import cv2
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import FashionMNIST
from torchvision import transforms


# 数据集划分
def FashionMNIST_split():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 将数据集划分为训练集和测试集
    X_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
    X_test, y_test = test_dataset.data.numpy(), test_dataset.targets.numpy()
    return X_train, X_test, y_train, y_test



# 定义特征提取函数

#像素值特征提取
def get_pixel_features():


    X_train, X_test, y_train, y_test = FashionMNIST_split()
    X_train_image = np.expand_dims(X_train, axis=-1)
    X_test_image = np.expand_dims(X_test, axis=-1)
    X_train_pixels = X_train.reshape(X_train_image.shape[0], -1)
    X_test_pixels = X_test.reshape(X_test_image.shape[0], -1)


    return X_train_pixels, X_test_pixels

#sift特征提取
def get_sift_features():
    X_train, X_test, _, _ = FashionMNIST_split()
    # 创建 SIFT 对象
    sift = cv2.SIFT_create()

    # 提取训练集和测试集的 SIFT 特征
    X_train_sift = []
    X_test_sift = []


    for i in range(len(X_train)):
        img = X_train[i].reshape(28, 28).astype(np.uint8)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:

            X_train_sift.append(torch.from_numpy(des).mean(dim=0))

        else:
            X_train_sift.append(torch.zeros(128))

    for i in range(len(X_test)):
        img = X_test[i].reshape(28, 28).astype(np.uint8)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:

            X_test_sift.append(torch.from_numpy(des).mean(dim=0))

        else:
            X_test_sift.append(torch.zeros(128))



    X_train_sift = torch.stack(X_train_sift, dim=0)
    X_test_sift = torch.stack(X_test_sift, dim=0)



    return X_train_sift, X_test_sift
