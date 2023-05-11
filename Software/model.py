import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import datetime

def read_data(file_path):
    """读取CSV文件"""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """预处理数据"""
    # 将波峰间隔时间列转换为列表
    data['波峰间隔时间'] = data['波峰间隔时间'].apply(lambda x: [float(i) for i in x.strip('[]').split()])

    # 将标签列转为数字编码
    label_map = {'急刹': 0, '点刹': 1}
    data['标签'] = data['标签'].map(label_map)

    # Pad the '波峰间隔时间' column separately
    padded_intervals = tf.keras.preprocessing.sequence.pad_sequences(data['波峰间隔时间'].tolist(), maxlen=10, dtype='float32', padding='post', truncating='post')

    # Combine the padded '波间隔时间' column with the other columns
    data[['信号持续时间', '波峰数量', '峰峰值']] = data[['信号持续时间', '波峰数量', '峰峰值']].to_numpy(dtype='float32')
    combined_data = np.hstack((data[['信号持续时间', '波峰数量', '峰峰值']].to_numpy(), padded_intervals))

    return combined_data

def split_data(data, train_ratio=0.8):
    """划分训练集测试集"""
    train_data = data[:int(len(data)*train_ratio)]
    test_data = data[int(len(data)*train_ratio):]
    return train_data, test_data

def create_model():
    """定义CNN模型"""
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(13, 1)),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model

def compile_model(model):
    """编译模型"""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def check_model_file(model, train_data, tensorboard_callback):
    """检查模型文件是否存在，如果存在则加载模型，否则训练模型并保存"""
    if os.path.exists('model/model.h5'):
        model.load_weights('model/model.h5')
    else:
        x_train = train_data[:, :, np.newaxis]
        y_train = train_data[:, -1].astype(int)
        model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])
        model.save_weights('model/model.h5')
    return model

def evaluate_model(model, test_data):
    """测试模型"""
    x_test = test_data[:, :, np.newaxis]
    y_test = test_data[:, -1].astype(int)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_loss, test_acc

def predict_labels(model, test_data, label_map):
    """输出预测结果"""
    x_test = test_data[:, :, np.newaxis]
    predictions = model.predict(x_test)
    predicted_labels = [list(label_map.keys())[list(label_map.values()).index(np.argmax(prediction))] for prediction in predictions]
    print("模型的预测结果：", predictions)
    print("标签映射：", label_map)
    return predicted_labels

def main():
    file_path = 'data/1.csv'
    data = read_data(file_path)
    combined_data = preprocess_data(data)
    train_data, test_data = split_data(combined_data)

    # 使用TensorBoard可视化模型
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = create_model()
    model = compile_model(model)
    model = check_model_file(model, train_data, tensorboard_callback)

    test_loss, test_acc = evaluate_model(model, test_data)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    label_map = {'急刹': 0, '点刹': 1}
    predicted_labels = predict_labels(model, test_data, label_map)
    print(predicted_labels)

    tf.keras.utils.plot_model(model, to_file='model/model.png', show_shapes=True)
    return

if __name__ == '__main__':
    main()
