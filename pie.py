import pandas as pd

# 加载数据
df = pd.read_csv('processed_data317.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

import matplotlib.pyplot as plt

# 对Wind Speed列进行分箱，划分成三个区间，绘制风速区间
df['Wind Speed Bins'] = pd.qcut(df['Wind Speed'], q=3)

# 统计每个区间的数量
bin_counts = df['Wind Speed Bins'].value_counts()

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 绘制饼状图
plt.pie(bin_counts, labels=bin_counts.index, autopct='%1.1f%%')
plt.title('风速区间分布饼图')
plt.show()

# 绘制湿度饼状图
import matplotlib.pyplot as plt

# 对 Humidity 列进行分箱操作，分为 5 个区间
df['Humidity_bin'] = pd.cut(df['Humidity'], bins=5)

# 统计每个区间的数量
humidity_counts = df['Humidity_bin'].value_counts()

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 绘制饼状图
plt.pie(humidity_counts, labels=humidity_counts.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('湿度区间分布饼图')
plt.show()
