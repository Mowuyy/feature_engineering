from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

"""
特征预处理应用场景：
        1、特征与特征之间单位或大小相差较大
        2、特征与特征之间的方差相差太大
        
"""


def minmax_demo():
    """
    特征归一化：
            x' =(x - min) / (max - min)
            x'' = x'(mx - mi) + mi  ==> mx、mi表示映射区间

            注意：1、只需对特征值进行归一化
                 2、作用于每一列
                 3、容易受异常点（不是最大值就是最小值）影响

    :return:
    """

    # 1、读取数据
    data = pd.read_csv('./data/dating.txt')
    print(data)

    # 2、实例化
    transfer = MinMaxScaler(feature_range=(2, 3))  # feature_range表示映射区间
    # 3、特征值转换
    new_data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print(new_data)

    return None


def standar_demo():
    """
    特征标准化：
            x' = (x - mean) / 标准差

            注意：1、只需对特征值进行标准化
                 2、作用于每一列
                 3、少量的异常点对均值影响不大，处理了异常点问题，稳定

    :return:
    """

    # 1、读取数据
    data = pd.read_csv('./data/dating.txt')
    print(data)

    # 2、实例化
    transfer = StandardScaler()
    # 3、特征值转换
    new_data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print('特征值标准化结果:\n', new_data)
    print('每列特征的均值:', transfer.mean_)
    print('每列特征的方差:', transfer.var_)

    return None


if __name__ == '__main__':
    # 1、特征归一化
    # minmax_demo()

    # 2、特征标准化
    standar_demo()
