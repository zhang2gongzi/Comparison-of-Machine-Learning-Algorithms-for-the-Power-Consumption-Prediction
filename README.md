﻿# Comparison-of-Machine-Learning-Algorithms-for-the-Power-Consumption-Prediction
参照论文Comparison of Machine Learning Algorithms for the Power Consumption Prediction:https://ieeexplore.ieee.org/document/8703007
数据集：https://www.kaggle.com/datasets/ruchikakumbhar/power-consumption-prediction-dataset
<img width="439" alt="fcf6b0456f1ac2875863538e11718fc" src="https://github.com/user-attachments/assets/96aaeada-42ab-4042-9eb0-a967ce162822" />
## 程序介绍
data-show.py:
这段 Python 代码主要实现了两个功能：绘制三个区域的功率消耗折线图以及绘制特征变量与目标变量之间的相关系数热力图：
1. 数据读取与折线图绘制
数据读取：
使用pandas库的read_csv函数从processed_data.csv文件中读取数据，将第一列解析为日期时间类型，并将其设置为索引列。
数据筛选：
筛选出Zone 1、Zone 2和Zone 3列中值小于 10000 的数据，并将这些值设置为NaN（缺失值）。
折线图绘制：
绘制Zone 1、Zone 2和Zone 3列数据的折线图，设置透明度为 0.7，颜色分别为绿色、橙色和蓝色。
图表设置：
设置图表标题为 “Power consumption for each hour in 2017 for three zones”。
设置 x 轴标签为 “Datetime”，y 轴标签为 “Power consumption (kW)”。
显示图例，图例内容为三个区域的名称。
最后使用plt.show()显示绘制好的折线图。
2. 数据处理与热力图绘制
数据再次读取：
再次使用pandas的read_csv函数从processed_data.csv文件中读取数据。
变量提取：
提取特征变量，包括Temperature、Humidity、Wind Speed、general diffuse flows和diffuse flows。
提取目标变量，即Zone 1、Zone 2和Zone 3。
相关系数计算：
计算特征变量和目标变量组合数据的相关系数矩阵。
从相关系数矩阵中提取特征变量与目标变量之间的相关系数，并打印输出。
热力图绘制：
使用seaborn库的heatmap函数绘制相关系数矩阵的热力图，设置图形大小为 10x8，显示数值注释，颜色映射为coolwarm。
设置热力图标题为 “Correlation Heatmap”。
最后使用plt.show()显示绘制好的热力图。
几个模型文件：
导入了以下必要的库
pandas：用于数据处理和分析，特别是读取 CSV 文件。
train_test_split：用于将数据集划分为训练集和测试集。
DecisionTreeRegressor：决策树回归模型，用于构建回归模型。
mean_squared_error 和 r2_score：用于评估模型的性能。
之后从processed_data.csv文件中读取数据。
提取特征变量（X）和目标变量（y）。
使用train_test_split函数将数据集划分为训练集（80%）和测试集（20%）
并设置随机种子为 42 以确保结果可复现。
https://blog.csdn.net/xiaohutong1991/article/details/107923970（随机种子作用）
创建一个随机森林回归模型，n_estimators 表示决策树的数量，设置为 100（rf.py,随机森林模型）。
使用训练好的模型对测试集进行预测。
计算目标变量的标准差，并根据标准差设定误差阈值。
遍历测试集，统计预测误差在阈值范围内的样本数，计算近似准确度。
输出均方误差、决定系数和近似准确度。
处理数据的程序，hour.py
该脚本的主要功能是读取一个包含电力消耗数据的 CSV 文件，对数据进行处理（每 6 行数据进行平均值计算），并在处理前后分别进行空值和异常值的检测，最后将处理后的数据保存到一个新的 CSV 文件中。

