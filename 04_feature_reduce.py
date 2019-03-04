from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import pandas as pd

"""
    特征降维：
        作用：减少特征数量，要求特征之间不相关
        
        分类：
            1、低方差特征过滤
            2、皮尔逊相关系数
            3、主成分分析（PCA）

"""


def variance_demo():
    """
    低方差过滤：对低于设定方差阈值范围的特征直接删除

    :return:
    """
    # 1、读取数据
    data = pd.read_csv('./data/factor_returns.csv')
    print(data)  # 观察数据，去掉特征index、date列
    print(data.shape)

    # 2、实例化
    transfer = VarianceThreshold(threshold=1)  # threshold表示设定的阈值（方差）
    new_data = transfer.fit_transform(data.iloc[:, 1:-2])
    print('低方差后结果:\n', new_data)
    print(new_data.shape)

    return None


def pearsonr_demo():
    """
    皮尔逊相关系数:
            两特征的 协方差 和 标准差 的商

            作用：
                1、判断特征与特征之间的相关性
                2、根据相关性再做后续处理，比如两特征是否合成一个特征、或者删除一个特征
    :return:
    """
    # 1、读取数据
    data = pd.read_csv('./data/factor_returns.csv')
    print(data)

    # 2、实例化
    r = pearsonr(data['pe_ratio'], data['pb_ratio'])
    print('皮尔逊相关系数: ', r[0])

    return None


def pca_demo():
    """
    主成分分析（PCA）：
                作用：减少特征的维度，对特征值进行压缩，损失少量的信息
    :return:
    """
    # 1、准备数据
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    print(data)
    # 2、实例化
    trasfer = PCA(n_components=0.95)  # n_components为小数时：表示保留百分之多少的信息，n_components为整数时：表示减少到多少特征
    new_data = trasfer.fit_transform(data)
    print(new_data)

    return None


if __name__ == '__main__':
    # 1、低方差过滤
    # variance_demo()

    # 2、皮尔逊相关系数
    # pearsonr_demo()

    # 3、PCA降维
    pca_demo()
