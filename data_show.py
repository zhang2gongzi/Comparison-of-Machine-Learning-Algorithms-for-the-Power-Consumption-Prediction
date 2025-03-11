import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 从CSV文件读取数据，假设文件名为data.csv，第一列为时间，后面三列为zone1, zone2, zone3
df = pd.read_csv('processed_data.csv', parse_dates=[0], index_col=0)

# 筛选出每个区域中值大于等于 10000 的数据，小于 10000 的值设为 NaN
df[df[['Zone 1', 'Zone 2', 'Zone 3']] < 10000] = float('nan')

# 绘制折线图，并指定颜色
df[['Zone 1', 'Zone 2', 'Zone 3']].plot(kind='line', alpha=0.7, color=['green', 'orange', 'blue'])

# 设置图表标题和轴标签
plt.title('Power consumption for each hour in 2017 for three zones')
plt.xlabel('Datetime')
plt.ylabel('Power consumption(kW)')

# 显示图例
plt.legend(['Zone 1', 'Zone 2', 'Zone 3'])

# 显示图表
plt.show()

# 读取 CSV 文件
data = pd.read_csv('processed_data.csv')

# 提取特征变量和目标变量
features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
targets = ['Zone 1', 'Zone 2', 'Zone 3']

# 计算相关系数矩阵
correlation_matrix = data[features + targets].corr()

# 提取特征与目标变量之间的相关系数
feature_target_correlation = correlation_matrix.loc[features, targets]

print(feature_target_correlation)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()