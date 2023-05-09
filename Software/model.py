import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

# 数据集导入与预处理
data_folder = 'Software/data/train_data'
file_list = os.listdir(data_folder)
data_frames = []

for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(data_folder, file)
        data_frames.append(pd.read_csv(file_path))

train_df = pd.concat(data_frames)
train_df_prep = train_df
scaler = MinMaxScaler()
train_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']] = scaler.fit_transform(train_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']])

# 划分训练集
X_train = train_df_prep.drop('label', axis=1).values
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))

le = LabelEncoder()
y_train = le.fit_transform(train_df_prep['label'])

model_file = 'Software/model/my_model.h5'

if os.path.exists(model_file):
    # 加载模型
    model = load_model(model_file)
else:
    # 创建RNN模型
    model = Sequential()
    model.add(LSTM(32, input_shape=(3, 1)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=200,
              batch_size=32,
              validation_split=0.2)
    # 保存模型
    model.save(model_file)

# 测试集导入与预处理
test_df = pd.read_csv('Software/data/test_data/test_data.csv')
test_df_prep = test_df
test_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']] = scaler.transform(test_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']])

# 划分测试集
X_test = test_df_prep.drop('label', axis=1).values
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
y_test = test_df_prep['label']

# 测试集检测
y_pred_prob = model.predict(X_test.reshape((X_test.shape[0], 3, 1)))
y_pred_rnn = np.argmax(y_pred_prob, axis=1)

# 输出结果
print('Test Acc rnn:', accuracy_score(y_test, y_pred_rnn))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred_rnn)

# 创建混淆矩阵的热图可视化
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - RNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
