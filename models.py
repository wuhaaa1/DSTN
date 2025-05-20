import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

# 定义常量
seq_len = 1


# 编码
class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):

        x = self.value_embedding(x) + self.position_embedding(x)
        x = self.dropout(x)


        return x

# SimMHSANet
class SimMHSANet(nn.Module):
    def __init__(self, seq_dim=4, seq_len=10, e_layers=2, n_heads=4, d_model=128, out_size=128, dropout=0.0, embed='fixed', freq='h'):
        super(SimMHSANet, self).__init__()

        self.seq_dim = seq_dim # 时间序列维度
        self.seq_len = seq_len # 时间序列长度
        self.e_layers = e_layers # 层数
        self.n_heads = n_heads #
        self.d_model = d_model #
        self.out_size = out_size # 输出维度
        self.dropout = dropout
        self.embed = embed
        self.freq = freq

        # Embedding
        self.enc_embedding = DataEmbedding(self.seq_dim, self.d_model, self.embed, self.freq, self.dropout)

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dropout=self.dropout)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.e_layers)

        self.fc = torch.nn.Linear(self.seq_len * self.d_model, self.out_size, bias=False)

    def forward_once(self, batch_x):
        #transform
        x_enc = torch.reshape(torch.as_tensor(batch_x), [-1, self.seq_dim, self.seq_len]).permute(0, 2, 1) # batch_x:[batch_size, seq_dim, seq_len]
        x_enc_emb =  self.enc_embedding(x_enc)  # input:[batch_size, seq_len, seq_dim]
        enc_out = self.encoder(x_enc_emb) # input:[batch_size, seq_len, d_model]
        output = self.fc(torch.reshape(enc_out, [-1, self.seq_len * self.d_model]))
        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)

        return (output1+output2)/2
    # transform 两次 输出值

class transform (nn.Module):
    def __init__(self, seq_dim=4, seq_len=10, e_layers=2, n_heads=4, d_model=128, out_size=128, dropout=0.0, embed='fixed', freq='h'):
        super(transform, self).__init__()

        self.seq_dim = seq_dim # 时间序列维度
        self.seq_len = seq_len # 时间序列长度
        self.e_layers = e_layers # 层数
        self.n_heads = n_heads #
        self.d_model = d_model #
        self.out_size = out_size # 输出维度
        self.dropout = dropout
        self.embed = embed
        self.freq = freq

        # Embedding
        self.enc_embedding = DataEmbedding(self.seq_dim, self.d_model, self.embed, self.freq, self.dropout)

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dropout=self.dropout)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.e_layers)

        self.fc = torch.nn.Linear(self.seq_len * self.d_model, self.out_size, bias=False)

    def forward_once(self, batch_x):
        # transform
        x_enc = torch.reshape(torch.as_tensor(batch_x), [-1, self.seq_dim, self.seq_len]).permute(0, 2,
                                                                                                  1)  # batch_x:[batch_size, seq_dim, seq_len]
        x_enc_emb = self.enc_embedding(x_enc)  # input:[batch_size, seq_len, seq_dim]
        enc_out = self.encoder(x_enc_emb)  # input:[batch_size, seq_len, d_model]
        output = self.fc(torch.reshape(enc_out, [-1, self.seq_len * self.d_model]))
        # 复合函数

        return output

    def forward(self, input1):
        output1 = self.forward_once(input1)
        return output1
      #transform 只掉一次 输出值


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class BPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=1):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def train_model(model, X_train, y_train, X_val, y_val,lr=0.001 , wd=0.001,epochs=100):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr ,weight_decay=wd)
    # 列表用于保存每个epoch的训练和验证损失
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # 训练模式
        model.train()
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        try: outputs = model(X_train)
        except TypeError:outputs = model(X_train,X_train)
        # 计算损失
        loss = criterion(outputs, y_train)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 保存训练损失
        train_losses.append(loss.item())
        # 验证模式
        model.eval()

        with torch.no_grad():
            try:val_outputs = model(X_val)
            except TypeError:val_outputs = model(X_val,X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return train_losses, val_losses

class IGWO:
    def __init__(self):
        # Initializing some class variables that will be used later
        self.Alpha_pos = None
        self.Beta_pos = None
        self.Delta_pos = None
        self.Alpha_score = float('-inf')
        self.Beta_score = float('-inf')
        self.Delta_score = float('-inf')

    @staticmethod
    def rand_generate(popsize, Encode):
        # generate solution
        population = np.zeros((popsize, sum(Encode['dnum'])))
        for j in range(Encode['degree']):
            # use the parameter's dimension generate solution
            buf = np.random.rand(popsize, Encode['dnum'][j])
            range_ = slice(sum(Encode['dnum'][:j]), sum(Encode['dnum'][:j + 1]))
            # check the dtype of parameters
            if Encode['style'][j] == 0:
                population[:, range_] = np.round(
                    buf * (Encode['bounds'][j][1] - Encode['bounds'][j][0]) + Encode['bounds'][j][0])
            elif Encode['style'][j] == 1:
                population[:, range_] = buf * (Encode['bounds'][j][1] - Encode['bounds'][j][0]) + Encode['bounds'][j][0]
            elif Encode['style'][j] == 2:
                sorted_indices = np.argsort(buf, axis=1)
                population[:, range_] = Encode['bounds'][j][sorted_indices]
        return population

    @staticmethod
    def cacul(popsize, population, data, fsolver, Encode):
        fnow = np.zeros(popsize)
        resultall = [None] * popsize
        for i in range(popsize):
            fnow[i], resultall[i] = fsolver['fitness'](population[i], data, Encode)
        fnow *= fsolver['minmax']
        fbest = np.max(fnow)
        fmean = np.mean(fnow)
        index = np.argmax(fnow)
        return fnow, fbest, fmean, resultall, index

    def main(self, para, Encode, fsolver, data):
        NP = para['popsize']
        max_Iter = para['Max_Iter']
        dim = Encode['dnum']

        X = self.rand_generate(NP, Encode)
        fnow, _, _, _, _ = self.cacul(NP, X, data, fsolver, Encode)
        pbest = X.copy()
        fitPb = fnow.copy()
        # every iteration fmean and fGbest occur
        V = 0.3 * np.random.randn(NP, sum(dim))
        fmean = np.zeros(max_Iter)
        fGbest = np.zeros(max_Iter)
        sbest = []
        iter_ = 0

        while iter_ < max_Iter:
            aa = np.max(X, axis=0)
            bb = np.min(X, axis=0)
            OX = np.random.rand(*X.shape) * (aa + bb) - X
            OXfnow, _, _, _, _ = self.cacul(NP, OX, data, fsolver, Encode)
            fnow, _, _, _, _ = self.cacul(NP, X, data, fsolver, Encode)

            mask = OXfnow < fnow
            X[mask] = OX[mask]
            fnow[mask] = OXfnow[mask]
            fitness = fnow

            for i in range(NP):
                if fitness[i] > self.Alpha_score:
                    self.Alpha_score = fitness[i]
                    self.Alpha_pos = X[i].copy()
                elif fitness[i] < self.Alpha_score and fitness[i] > self.Beta_score:
                    self.Beta_score = fitness[i]
                    self.Beta_pos = X[i].copy()
                elif fitness[i] < self.Alpha_score and fitness[i] < self.Beta_score and fitness[i] > self.Delta_score:
                    self.Delta_score = fitness[i]
                    self.Delta_pos = X[i].copy()

                if fitPb[i] < fnow[i]:
                    pbest[i] = X[i].copy()
                    fitPb[i] = fnow[i]

            a_final = 0
            a_ini = 2
            n = 1
            if iter_ < 0.5 * max_Iter:
                a = a_final + (a_ini - a_final) * (1 + np.cos(iter_ * np.pi / max_Iter) ** n) / 2
            else:
                a = a_final + (a_ini - a_final) * (1 - np.cos(iter_ * np.pi / max_Iter) ** n) / 2

            # Update the Position of search agents including omegas
            for i in range(NP):
                for j in range(sum(dim)):
                    A1, C1 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    D_alpha = abs(C1 * self.Alpha_pos[j] - X[i, j])
                    X1 = self.Alpha_pos[j] - A1 * D_alpha

                    A2, C2 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    D_beta = abs(C2 * self.Beta_pos[j] - X[i, j])
                    X2 = self.Beta_pos[j] - A2 * D_beta

                    A3, C3 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    D_delta = abs(C3 * self.Delta_pos[j] - X[i, j])
                    X3 = self.Delta_pos[j] - A3 * D_delta

                    X[i, j] = (X1 + X2 + X3) / 3
                    r3, r4 = np.random.randint(0, NP, 2)
                    while r3 == i:
                        r3 = np.random.randint(0, NP)
                    while r4 == i or r3 == r4:
                        r4 = np.random.randint(0, NP)
                    X[i, j] += 0.5 * (X[r3, j] - X[r4, j])

            fGbest[iter_] = self.Alpha_score
            fmean[iter_] = np.mean(fnow)
            sbest.append(self.Alpha_pos)
            iter_ += 1

        results = fsolver['fitness'](self.Alpha_pos, data, Encode)[0]
        sGbest = sbest[np.argmax(fGbest)]
        paintd = {
            'f_best': fGbest[:iter_] * fsolver['minmax'],
            'f_fitness': fGbest[:iter_],
            'f_fitmean': fmean[:iter_],
            'gb': fGbest[iter_ - 1]
        }
        return results, paintd

def get_train_data(X_train,X_val,y_val,y_train):
    train_dataset = TensorDataset(X_train,y_train)
    val_dataset  = TensorDataset(X_val,y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return (train_loader,val_loader)

def train_network(hyperparameters, data=None,encode=None,epochs=10):
    # Extract hyperparameters
    learning_rate,weight_decay = adjust_hyperparameters(hyperparameters[0],hyperparameters[1],encode['bounds'])
    train_loader,val_loader = get_train_data(data['X_train'],data['X_val'],data['y_val'],data['y_train'])
    # Initialize network and optimizer
    model = SimMHSANet(seq_dim=data['X_train'].shape[1], seq_len=seq_len, out_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data,data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

    # Validation loop
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            outputs = model(data,data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, None  # Return validation loss as fitness value


def adjust_hyperparameters(learning_rate, weight_decay, bounds):
    """enhance use bounds check bounds in a loop"""
    learning_rate_diff = abs(bounds[0][1][0]-bounds[0][0][0])
    weight_decay_diff = abs(bounds[1][1][0]-bounds[1][0][0])
    # 检测learning_rate是否在边界内
    if learning_rate < bounds[0][0][0]:
        learning_rate = bounds[0][0][0] +.01*learning_rate_diff
    elif learning_rate > bounds[0][1][0]:
        learning_rate = bounds[0][1][0] -.01*learning_rate_diff

    # 检测weight_decay是否在边界内
    if weight_decay < bounds[1][0][0]:
        weight_decay = bounds[1][0][0] +.01*weight_decay_diff
    elif weight_decay > bounds[1][1][0]:
        weight_decay = bounds[1][1][0] -.01*weight_decay_diff

    return learning_rate, weight_decay
