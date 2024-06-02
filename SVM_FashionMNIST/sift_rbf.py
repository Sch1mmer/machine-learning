import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from SVM_FashionMNIST import get_sift_features, FashionMNIST_split

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']




print('--------SIFT特征+rbf核函数--------')

# 提取特征
print('特征提取...')
X_train, X_test = get_sift_features()
_, _, y_train, y_test = FashionMNIST_split()

# 训练模型
print('模型训练...')
model = SVC(kernel='rbf', C=50, gamma=0.1)
now = datetime.datetime.now()
formatted_time = now.strftime("%H:%M:%S")
print(formatted_time)
model.fit(X_train, y_train)



# 评估模型
print('模型评估...')
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"训练准确率: {train_accuracy:.4f}")
print(f"测试准确率: {test_accuracy:.4f}")


# 混淆矩阵
y_true = y_test  # 真实标签
y_pred = model.predict(X_test)  # 预测标签
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm = confusion_matrix(y_true, y_pred)
# 设置图片大小
plt.figure(figsize=(10, 8))

# 绘制混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('混淆矩阵（siftl+rbf核函数）')
plt.colorbar()

# 添加坐标轴标签
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=90)
plt.yticks(tick_marks, class_labels)

# 添加数值标注
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

# 调整图片边距
plt.tight_layout()
plt.ylabel('真实标签')
plt.xlabel('预测标签')

# 保存图片
plt.savefig(f"data/混淆矩阵(sift_rbf)_C100_test.png")
plt.close()