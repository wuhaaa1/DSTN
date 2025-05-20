# 首先进行数据的可视化visualize
import os
from preprocess import *
from models import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch


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


seq_len = 1

# 导入模型 # 开始训练
models = [
    ("DS-SAM",SimMHSANet(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1),X_train_reshaped,X_val_reshaped),
    #("LSTM", LSTMNet(input_dim=1, hidden_dim=50), X_train_reshaped, X_val_reshaped),
    ("BPNet", BPNet(input_dim=X_train.shape[1], hidden_dim=50), X_train, X_val),
    ("transform",transform(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1),X_train_reshaped,X_val_reshaped)


]


loss_history = []
colors = ['blue','green','yellow','red']
for (name, model, X_train, X_val),c in zip(models,colors):
    loss_dict={}
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val,epochs=100)
    loss_dict["train_loss"] = train_losses
    loss_dict["val_loss"] = val_losses
    loss_history.append({name:loss_dict})
    plt.plot(train_losses,color=c,label=f'{name} Training Loss')
    plt.plot(val_losses,color=c,marker='o',label=f'{name} Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Comparison of Training and Validation Losses for Different Models')
plt.show()



# 开始测试
model = models[0][1]
model.eval()
# reshape the value and get the origin value
X_test = X_test.values
X_test_reshaped = X_test.reshape(-1, X_test.shape[1], seq_len)#.values
# convert to tensor
X_test_reshaped_tensor = torch.Tensor(X_test_reshaped)
X_test_tensor = torch.Tensor(X_test)

# 使用模型进行预测
with torch.no_grad():
    try:y_pred = model(X_test_tensor)
    except TypeError: y_pred = model(X_test_tensor,X_test_tensor)
    finally:
      y_pred =torch.flatten(y_pred)

# 对于回归问题，计算均方误差（MSE）来评估模型性能
mse = mean_squared_error(y_test.values, y_pred.detach().numpy())
print(f'Mean Squared Error: {mse:.4f}')
plt.figure()
plt.plot(y_pred.detach().numpy(),label='predicted')
plt.plot(y_test.values,label='true')
plt.legend()
plt.show()


# 消融实验
model_list = [SimMHSANet(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1,dropout=.1),
        SimMHSANet(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1,dropout=.2),
        SimMHSANet(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1,dropout=.3)]
plt.figure()
ablation_history = {}
colors = ['blue','green','yellow']
# train the model
for k,sim_model in enumerate(model_list,start=1):
    train_loss,val_loss = train_model(sim_model,X_train_reshaped,y_train,X_val_reshaped,y_val)
    plt.plot(train_loss,color=colors[k-1],label=f'DS-SAM train dropout=0.{k}')
    plt.plot(val_loss,color=colors[k-1],label=f'DS-SAM val dropout=0.{k}')
    ablation_history[f'model dropout=0.{k} train_loss'] = train_loss
    ablation_history[f'model dropout=0.{k} val_loss'] = val_loss
plt.legend()
plt.show()

# 消融实验测试
loss_dict_dropout={'true-value':y_test.values}
plt.figure()
for label_name,model in zip(['dropout=0.1','dropout=0.2','dropout=0.3'],model_list):
    with torch.no_grad():
      try:y_pred = model(X_test_tensor)
      except TypeError: y_pred = model(X_test_tensor,X_test_tensor)
      finally:
        #
        y_pred =torch.flatten(y_pred)

    # 对于回归问题，计算均方误差（MSE）来评估模型性能
    mse = mean_squared_error(y_test.values, y_pred.detach().numpy())
    print(f'{label_name} Mean Squared Error: {mse:.4f}')
    plt.plot(y_pred.detach().numpy(),label=f'{label_name} predicted')
    loss_dict_dropout[f'DS-SAM {label_name} '] = y_pred


# IGWO优化 - 参数设定 求取最优参数
fsolver = {
    'fitness': train_network,
    'minmax': -1,
}
Encode = {
    'degree': 2,
    'dnum': [1, 1],
    'style': [1, 1],
    'bounds': [np.array([[0.001], [0.1]]), np.array([[0], [0.01]])]

}
# bound 为 lr wd的边界 dnum为各自超参的维度  degree为整体超参个数
para = {
    'popsize': 10, # 10
    'Max_Iter': 20 # 20
}
data ={
    'X_train': X_train_reshaped,
    'X_val': X_val_reshaped,
    'y_val':y_val,
    'y_train':y_train,
}

optimizer = IGWO()
results, paintd = optimizer.main(para, Encode, fsolver, data)
print(results, paintd['gb'])
print(paintd)
print(optimizer.Alpha_pos)


# IGWO实验对比
lr,wd = optimizer.Alpha_pos

# wd 不能为负值
wd = .001 if wd < 0 else wd
model1 = models[0][1] #mdoel1 = models[0][1]
model2 = SimMHSANet(seq_dim=X_train.shape[1], seq_len=seq_len, out_size=1)
# use the best parameters to train the model
m2_train_loss,m2_val_loss = train_model(model2,X_train_reshaped,y_train,X_val_reshaped,y_val,lr=lr,wd=wd)
m1_train_loss,m1_val_loss = loss_history[0]['DS-SAM'].values()
bp_train_loss,bp_val_loss = loss_history[1]['BPNet'].values()

# draw the loss function effect
epochs = range(1, 10 + 1)

# plot each model effect
plt.figure()
plt.plot(epochs, m2_train_loss[:10], label='M2 Train Loss', linestyle='--', marker='o')
plt.plot(epochs, m2_val_loss[:10], label='M2 Validation Loss', linestyle='-', marker='x')
plt.plot(epochs, m1_train_loss[:10], label='M1 Train Loss', linestyle='--', marker='o')
plt.plot(epochs, m1_val_loss[:10], label='M1 Validation Loss', linestyle='-', marker='x')
plt.plot(epochs, bp_train_loss[:10], label='BPNet Train Loss', linestyle='--', marker='o')
plt.plot(epochs, bp_val_loss[:10], label='BPNet Validation Loss', linestyle='-', marker='x')
plt.legend()

# define the history variable
plt.title('Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# use test set to test the model
# 使用模型进行预测
loss_dict2={'true-value':y_test.values}
plt.figure()
for label_name,model in zip(['origin','optimize'],[model1,model2]):
    with torch.no_grad():
      try:y_pred = model(X_test_tensor)
      except TypeError: y_pred = model(X_test_tensor,X_test_tensor)
      finally:
        y_pred =torch.flatten(y_pred)

    # 对于回归问题，计算均方误差（MSE）来评估模型性能
    mse = mean_squared_error(y_test.values, y_pred.detach().numpy())
    print(f'{label_name} Mean Squared Error: {mse:.4f}')
    plt.plot(y_pred.detach().numpy(),label=f'{label_name} predicted')
    loss_dict2[f'{label_name} DS-SAM'] = y_pred
plt.plot(y_test.values,label='true')
plt.legend()
plt.show()


# use test set to test the model
# 使用模型进行预测
loss_dict2={'true-value':y_test.values}
colors = ['blue','green','yellow','red']
for (name, model, X_train, X_val),c in zip(models,colors):
    with torch.no_grad():
      try:y_pred = model(X_test_tensor)
      except TypeError: y_pred = model(X_test_tensor,X_test_tensor)
      finally:
        y_pred =torch.flatten(y_pred)

    # 对于回归问题，计算均方误差（MSE）来评估模型性能
    mse = mean_squared_error(y_test.values, y_pred.detach().numpy())
    print(f'{name} Mean Squared Error: {mse:.4f}')
    plt.plot(y_pred.detach().numpy(),label=f'{name} predicted')
    loss_dict2[f'{name} DS-SAM'] = y_pred
plt.plot(y_test.values,label='true')
plt.legend()
plt.show()
