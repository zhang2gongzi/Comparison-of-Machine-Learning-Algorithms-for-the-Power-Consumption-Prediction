import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 定义 CSV 文件的路径
file_path = 'processed_data.csv'

# 读取 CSV 文件并存储为 DataFrame 对象
df = pd.read_csv(file_path)
# 将DateTime列转换为日期时间类型，使用mixed参数自动推断格式
df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 1. Temperature随时间变化的折线图
plt.figure(figsize=(12, 6))
plt.plot(df['DateTime'], df['Temperature'])
plt.title('温度随时间变化')
plt.xlabel('日期时间')
plt.xticks(rotation=45)
plt.ylabel('温度')
plt.show()

# 2. Humidity的箱线图
plt.figure(figsize=(8, 6))
df['Humidity'].plot.box()
plt.title('湿度箱线图')
plt.ylabel('湿度')
plt.show()

# 3. Temperature、Humidity和Wind Speed之间的散点图矩阵
g = sns.pairplot(df[['Temperature', 'Humidity', 'Wind Speed']])
g.fig.suptitle('温度、湿度和风速的散点图矩阵', y=1.02)
plt.show()

