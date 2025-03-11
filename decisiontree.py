import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，你也可以选择其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取CSV文件
data = pd.read_csv('processed_data.csv')

# 提取特征变量和目标变量
X = data[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']]
y = data[['Zone 1', 'Zone 2', 'Zone 3']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归模型
model = DecisionTreeRegressor(random_state=42)

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

# 手动输入特征变量进行验证
print("\n请输入用于预测的特征变量：")
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
wind_speed = float(input("Wind Speed: "))
general_diffuse_flows = float(input("general diffuse flows: "))
diffuse_flows = float(input("diffuse flows: "))

# 创建新的输入数据
new_input = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Wind Speed': [wind_speed],
    'general diffuse flows': [general_diffuse_flows],
    'diffuse flows': [diffuse_flows]
})

# 使用训练好的模型进行预测
new_prediction = model.predict(new_input)

# 输出预测结果
print("\n预测的各zone数值如下：")
print(f"Zone 1: {new_prediction[0][0]}")
print(f"Zone 2: {new_prediction[0][1]}")
print(f"Zone 3: {new_prediction[0][2]}")

# 提取 Zone 1 的真实值和预测值
zone1_true = y_test['Zone 1'].values
zone1_pred = y_pred[:, 0]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(zone1_true, label='Zone 1 真实值', color='blue')
plt.plot(zone1_pred, label='Zone 1 预测值', color='red')
plt.title('Zone 1 用电量真实值与预测值对比')
plt.xlabel('样本编号')
plt.ylabel('用电量')
plt.legend()
plt.grid(True)
plt.show()