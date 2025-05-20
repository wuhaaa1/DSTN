import os
from preprocess import *
from models import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.svm import SVR

os.system('python visualize.py')

# 进行数据的预处理 preprocess
X_train,X_val,X_test,X_train_reshaped,X_val_reshaped,y_train,y_val,y_test = preprocess_data()
X_train_reshaped = torch.Tensor(X_train_reshaped)
X_train = torch.Tensor(X_train.values)
X_val_reshaped = torch.Tensor(X_val_reshaped)
X_val = torch.Tensor(X_val.values)

y_train = torch.unsqueeze(torch.Tensor(y_train.values),dim=1)
y_val = torch.unsqueeze(torch.Tensor(y_val.values),dim=1)

print(X_train.shape,X_train_reshaped.shape)

# 标准化训练集和测试集
sc = StandardScaler()   # 定义一个标准缩放器
sc.fit(X_train)         # 计算均值、标准差
X_train_std = sc.transform(X_train) # 使用计算出的均值和标准差进行标准化
X_test_std  = sc.transform(X_test)  # 使用计算出的均值和标准差进行标准化

# ? 训练线性支持向量机
svm = SVR(kernel='linear', C=1.0)  # 定义线性支持向量分类器 (linear为线性核函数)
svm.fit(X_train_std, y_train)  # 根据给定的训练数据拟合训练SVM模型

#? 使用测试集进行数据预测
y_pred = svm.predict(X_test_std)    # 用训练好的分类器svm预测数据X_test_std的标签
print('Misclassified samples: %d' % (y_test != y_pred).sum())   # 输出错误分类的样本数
print('Accuracy: %.2f' % svm.score(X_test_std, y_test))         # 输出分类准确率

print(y_pred.shape)
print(y_test.shape)
print(y_pred)
print(y_test)
print(y_val.shape)

mse = mean_squared_error(y_test.values, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
plt.plot(y_pred, label=f'predicted')
plt.plot(y_test.to_list(), label=f'predicted')
plt.show()

print(y_val.shape)