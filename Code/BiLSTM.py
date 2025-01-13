import akshare as ak
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


def get_stock_data(symbol, period, start_date, end_date):
    """
    使用akshare获取股票数据
    :param symbol: 股票代码
    :param period: 时间周期，如'daily'表示日线
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 股票收盘价格数据（numpy数组格式，形状为(-1, 1)，数据类型为float32）
    """
    stock_data = ak.stock_zh_a_hist(symbol, period, start_date, end_date)
    close_prices = stock_data['收盘'].values.reshape(-1, 1).astype('float32')
    return close_prices


def normalize_data(close_prices):
    """
    对股票价格数据进行归一化处理
    :param close_prices: 原始股票收盘价格数据（numpy数组）
    :return: 归一化后的价格数据和对应的MinMaxScaler对象（用于后续反归一化）
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    return scaled_prices, scaler


def split_data(scaled_prices):
    """
    划分训练集和测试集，按80%训练，20%测试的比例划分
    :param scaled_prices: 归一化后的股票价格数据（numpy数组）
    :return: 划分好的训练集数据和测试集数据（numpy数组格式）
    """
    train_size = int(len(scaled_prices) * 0.8)
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]
    return train_data, test_data


def construct_sequences(data, sequence_length):
    """
    构造输入输出序列，用于模型训练和测试
    :param data: 输入的数据集（训练集或测试集数据，numpy数组格式）
    :param sequence_length: 输入序列的长度，即利用过去多少天的数据预测下一天价格
    :return: 构造好的输入特征序列X和对应的输出目标序列y（numpy数组格式）
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def convert_to_tensors(X_train, y_train, X_test, y_test):
    """
    将输入输出序列转换为torch张量
    :param X_train: 训练集输入特征序列（numpy数组格式）
    :param y_train: 训练集输出目标序列（numpy数组格式）
    :param X_test: 测试集输入特征序列（numpy数组格式）
    :param y_test: 测试集输出目标序列（numpy数组格式）
    :return: 转换后的训练集和测试集的输入输出张量（torch张量格式）
    """
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return X_train, y_train, X_test, y_test


def bilstm_model(input_size=1, hidden_size=32, num_layers=2):
    """
    定义双向LSTMStockPredictor模型
    :param input_size: 输入数据的特征数量，默认为1
    :param hidden_size: LSTM隐藏层神经元数量，默认为32
    :param num_layers: LSTM的层数，默认为2
    :return: 实例化后的双向LSTMStockPredictor模型对象
    """
    class BiLSTMStockPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=2):
            super(BiLSTMStockPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, 1)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.bilstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    return BiLSTMStockPredictor(input_size, hidden_size, num_layers)


def train_model(model, X_train, y_train, epochs, lr):
    """
    训练模型
    :param model: 要训练的模型对象（双向LSTMStockPredictor实例）
    :param X_train: 训练集输入特征张量（torch张量格式）
    :param y_train: 训练集输出目标张量（torch张量格式）
    :param epochs: 训练轮数，默认为1000轮
    :param lr: 学习率，默认为0.001
    :return: 训练后的模型对象
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 添加早停相关参数
    patience = 20  # 容忍的轮数，即测试集性能多久没提升就停止训练
    min_delta = 0.0001  # 最小的性能提升幅度，小于这个幅度不算性能提升
    best_mse = float('inf')  # 初始最佳MSE设为无穷大
    best_epoch = 0  # 记录最佳性能对应的轮数

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

            # 在测试集上评估模型，计算MSE
            with torch.no_grad():
                test_predictions = model(X_test)
            test_mse = criterion(test_predictions, y_test).item()

            # 判断是否满足早停条件
            if test_mse < best_mse - min_delta:
                best_mse = test_mse
                best_epoch = epoch
            elif epoch - best_epoch > patience:
                break

    print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
    return model


def evaluate_model(model, X_test, y_test, scaler):
    """
    对模型进行评估，计算均方误差（MSE）、平均绝对误差（MAE）、平均绝对百分比误差（MAPE）和决定系数（R2）
    :param model: 训练好的模型对象（双向LSTMStockPredictor实例）
    :param X_test: 测试集输入特征张量（torch张量格式）
    :param y_test: 测试集输出目标张量（torch张量格式）
    :param scaler: 用于数据归一化的MinMaxScaler对象（用于反归一化结果）
    :return: MSE、MAE、MAPE、R2这几个评估指标的值
    """
    with torch.no_grad():
        test_predictions = model(X_test)

    test_predictions = scaler.inverse_transform(test_predictions.numpy())
    y_test_original = scaler.inverse_transform(y_test.numpy())

    mse = np.mean((test_predictions - y_test_original) ** 2)
    mae = np.mean(np.abs(test_predictions - y_test_original))
    mape = np.mean(np.abs((y_test_original - test_predictions) / y_test_original)) * 100
    r2 = r2_score(y_test_original, test_predictions)
    return mse, mae, mape, r2


def plot_predictions(y_test_original, test_predictions):
    """
    绘制预测结果图，展示真实价格和预测价格的对比
    :param y_test_original: 原始的测试集真实价格数据（反归一化后，numpy数组格式）
    :param test_predictions: 模型预测的价格数据（反归一化后，numpy数组格式）
    """
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(y_test_original, label='真实价格')
    plt.plot(test_predictions, label='预测价格')
    plt.xlabel('时间')
    plt.ylabel('股票价格')
    plt.title('BiLSTM股票预测')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 获取股票数据 总数7824条 时间1991-04-03至2023-12-29
    close_prices = get_stock_data(symbol='000001', period='daily', start_date='19910403', end_date='20231229')
    # 数据归一化
    scaled_prices, scaler = normalize_data(close_prices)
    # 划分训练集和测试集
    train_data, test_data = split_data(scaled_prices)
    # 构造训练集和测试集的输入输出序列（假设以过去50天预测下一天价格）
    sequence_length = 150
    X_train, y_train = construct_sequences(train_data, sequence_length)
    X_test, y_test = construct_sequences(test_data, sequence_length)
    # 转换为张量
    X_train, y_train, X_test, y_test = convert_to_tensors(X_train, y_train, X_test, y_test)
    # 定义模型
    model = bilstm_model()
    # 训练模型
    trained_model = train_model(model, X_train, y_train, epochs=1000, lr=0.001)
    # 评估模型
    mse, mae, mape, r2 = evaluate_model(trained_model, X_test, y_test, scaler)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}%')
    print(f'R-squared: {r2}')
    # 绘制预测图
    plot_predictions(scaler.inverse_transform(y_test.numpy()),
                     scaler.inverse_transform(trained_model(X_test).detach().numpy()))