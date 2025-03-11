import pandas as pd
from scipy.stats import zscore

# 输入和输出文件路径
input_csv = 'power_consumption.csv'  # 输入文件路径
output_csv = 'processed_data.csv'   # 输出文件路径

# 读取CSV文件
df = pd.read_csv(input_csv)

# 处理前的空值检测
null_counts_before = df.isnull().sum()
print("处理前各列的空值数量：")
print(null_counts_before)

# 处理前的异常值检测
if len(df.columns) > 1:
    z_scores_before = df.iloc[:, 1:].apply(zscore)
    outliers_before = (z_scores_before.abs() > 3).any(axis=1)
    outlier_count_before = outliers_before.sum()
    print(f"处理前检测到的异常值数量：{outlier_count_before}")
else:
    print("数据只有一列，无法进行异常值检测。")

# 假设第一列是时间戳，不参与平均值计算
# 从第二列开始，每6行数据进行平均值计算
group_size = 6
grouped_data = []

# 遍历数据，每次处理6行
for i in range(0, len(df), group_size):
    group = df.iloc[i:i + group_size]  # 获取每6行数据
    time_stamp = group.iloc[0, 0]  # 获取当前组的时间戳（第一列的第一个值）
    group_mean = group.mean(numeric_only=True)  # 计算数值列的平均值
    group_mean[df.columns[0]] = time_stamp  # 将时间戳添加到平均值结果中
    grouped_data.append(group_mean)  # 将结果添加到列表中

# 将结果转换为DataFrame
result_df = pd.DataFrame(grouped_data)

# 处理后的空值检测
null_counts_after = result_df.isnull().sum()
print("处理后各列的空值数量：")
print(null_counts_after)

# 处理后的异常值检测
if len(result_df.columns) > 1:
    z_scores_after = result_df.iloc[:, 1:].apply(zscore)
    outliers_after = (z_scores_after.abs() > 3).any(axis=1)
    outlier_count_after = outliers_after.sum()
    print(f"处理后检测到的异常值数量：{outlier_count_after}")
else:
    print("处理后的数据只有一列，无法进行异常值检测。")

# 保存到新的CSV文件
result_df.to_csv(output_csv, index=False)

print(f"处理完成，结果已保存到 {output_csv}")