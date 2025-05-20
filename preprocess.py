# 封装成为函数 之后使用各个变量进行后续的训练工作

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name,parameters in model.named_parameters():
      print(name,':',parameters.size())
    return {'Total': total_num, 'Trainable': trainable_num}

def preprocess_data():

    data = pd.read_excel('（全部白夜班）final_ori_data.xlsx')
    data.drop(columns=['Unnamed: 0', '日期','工作平盘宽度'], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['班次','车型']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Check for missing values
    missing_values = data.isnull().sum()

    # interpreter the columns drop the columns '里程 km','实际加油量 L'
    data.dropna(subset=['里程 km'],inplace=True)
    data['实际加油量 L'] = data['实际加油量 L'].interpolate(method='spline',order=3)


    for i in data.columns:
        data[i] = pd.to_numeric(data[i], errors='coerce')
    # Standardize numerical features
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


    data_scaled = data_scaled.dropna()
    # Assuming the last column is the target variable
    X = data_scaled.drop(columns=['吨公里油耗 L/km/t'])
    y = data_scaled['吨公里油耗 L/km/t']

    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further splitting the training set into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # Reshape the data to match the model's input shape [batch_size, seq_dim, seq_len]
    seq_len = 1

    X_train_reshaped = X_train.values.reshape(-1, X_train.shape[1], seq_len)
    X_val_reshaped = X_val.values.reshape(-1, X_val.shape[1], seq_len)

    return X_train,X_val,X_test,X_train_reshaped,X_val_reshaped,y_train,y_val,y_test

