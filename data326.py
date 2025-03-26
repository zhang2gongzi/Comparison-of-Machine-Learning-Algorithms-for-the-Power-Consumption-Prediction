import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，你也可以选择其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_variable_vs_zone(data, variable):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制与Zone 1的关系图
    sns.scatterplot(data=data, x=variable, y='Zone 1', ax=axes[0])
    axes[0].set_title(f'{variable}与Zone 1用电量的关系')
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel('Zone 1用电量')

    # 绘制与Zone 2的关系图
    sns.scatterplot(data=data, x=variable, y='Zone 2', ax=axes[1])
    axes[1].set_title(f'{variable}与Zone 2用电量的关系')
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel('Zone 2用电量')

    # 绘制与Zone 3的关系图
    sns.scatterplot(data=data, x=variable, y='Zone 3', ax=axes[2])
    axes[2].set_title(f'{variable}与Zone 3用电量的关系')
    axes[2].set_xlabel(variable)
    axes[2].set_ylabel('Zone 3用电量')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('processed_data317.csv')
    variables = ['Temperature', 'general diffuse flows', 'diffuse flows']
    for var in variables:
        plot_variable_vs_zone(data, var)
