import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取CSV文件
data = pd.read_csv('processed_data.csv')

# 提取特征变量和目标变量
X = data[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']]
y = data[['Zone 1', 'Zone 2', 'Zone 3']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多元线性回归模型
model = LinearRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算目标变量的标准差
std_dev = y_test.values.std()

# 根据标准差设定误差阈值
threshold = std_dev * 0.5  # 可根据实际情况调整

# 计算预测正确的样本数
correct_count = 0
total_count = len(y_test)

for i in range(total_count):
    if all(abs(y_pred[i] - y_test.iloc[i].values) <= threshold):
        correct_count += 1

# 计算近似准确度
accuracy = correct_count / total_count

print('均方误差：', mean_squared_error(y_test, y_pred))
print('决定系数：', r2_score(y_test, y_pred))
print('近似准确度：', accuracy)