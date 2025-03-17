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
#三个区域的随时间变化折线图，具体代码在data_show。py df[['Zone 1', 'Zone 2', 'Zone 3']].plot(kind='line', alpha=0.7, color=['green', 'orange', 'blue'])


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

#4.绘制不同小时下Zone 1、Zone 2和Zone 3的箱线图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxplots_for_zones(csv_file_path):
    # 加载数据
    df = pd.read_csv(csv_file_path)

    # 将 DateTime 列转换为日期时间类型，使用 mixed 参数来推断每个元素的日期时间格式
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')

    # 提取小时信息
    df['Hour'] = df['DateTime'].dt.hour

    # 将数据从宽格式转换为长格式
    melted_df = pd.melt(df, id_vars=['Hour'], value_vars=['Zone 1', 'Zone 2', 'Zone 3'], var_name='Zone', value_name='Value')

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

    # 绘制不同小时下不同区域的箱线图
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(x='Hour', y='Value', hue='Zone', data=melted_df)

    # 添加图标题和轴标签
    plt.title('不同小时下不同区域的箱线图')
    plt.xlabel('小时')
    plt.xticks(rotation=45)
    plt.ylabel('数值')
    plt.legend(title='区域')

    # 显示图形
    plt.show()

# 调用函数并传入你的 CSV 文件路径
# 这里使用你上传文件的路径示例，实际使用时如果路径不同需修改
file_path = 'processed_data.csv'
plot_boxplots_for_zones(file_path)

