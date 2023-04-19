from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM

gnb = GaussianNB()

# 极大似然估计
def max_likelihood(x, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    p1 = norm.pdf(x, mu1, sigma1)
    p2 = norm.pdf(x, mu2, sigma2)
    p3 = norm.pdf(x, mu3, sigma3)
    p = p1 + p2 + p3
    return np.argmax([p1/p, p2/p, p3/p])

# accuracy_score函数
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

# 数据集导入与预处理
train_df = pd.read_csv('Software/data/train_data.csv')
train_df_prep = train_df
scaler = MinMaxScaler()
train_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']] = scaler.fit_transform(train_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']])

# 划分训练集
X_train = train_df_prep.drop('label', axis=1).values
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))

le = LabelEncoder()
y_train = le.fit_transform(train_df_prep['label'])

# 载入模型
mu1, sigma1 = norm.fit(train_df[y_train==0])
mu2, sigma2 = norm.fit(train_df[y_train==1])
mu3, sigma3 = norm.fit(train_df[y_train==2])
gnb.fit(X_train, y_train)

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

# 测试集导入与预处理
test_df = pd.read_csv('Software/data/test_data.csv')
test_df_prep = test_df
test_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']] = scaler.transform(test_df_prep[['Signal Strength', 'Press Speed', 'Press Duration']])

# 划分测试集
X_test = test_df_prep.drop('label', axis=1).values
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
y_test = test_df_prep['label']

# 测试集检测
y_pred_like = [max_likelihood(x, mu1, sigma1, mu2, sigma2, mu3, sigma3) for x in X_test]
y_pred_bayes = gnb.predict(X_test)
y_pred_prob = model.predict(X_test.reshape((X_test.shape[0], 3, 1)))
y_pred_rnn = np.argmax(y_pred_prob, axis=1)

# 输出结果
print('Test Acc like:', accuracy_score(y_test, y_pred_like))
print('Test Acc bayes:', accuracy_score(y_test, y_pred_bayes))
print('Test Acc rnn:', accuracy_score(y_test, y_pred_rnn))
