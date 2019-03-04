from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets_demo():
    """
    数据集划分：训练集和测试集分离
    :return:
    """
    iris = load_iris()

    print(dir(iris))
    print('特征名称:\n', iris.feature_names)
    print('特征值:\n', iris.data)
    print('目标名称:\n', iris.target_names)
    print('目标值:\n', iris.target)

    # 特征值和测试值分离（数据打乱）
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=20)
    # random_state: 表示确保每次运行的随机结果值一样
    # test_size: 表示测试值所占比例

    print('=' * 100)
    print('训练特征值:\n', x_train)
    print('测试特征值:\n', x_test)
    print('训练目标值:\n', y_train)
    print('测试目标值:\n', y_test)

    return None


if __name__ == '__main__':
    datasets_demo()
