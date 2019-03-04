from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

"""
特征抽取应用场景：
    1、字典特征抽取（构造好如下键值对数据）
    2、文本特征抽取（英文,中文jieba分词、tfidf）
    3、图片特征抽取（深度学习）
    
"""


def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    transfer = DictVectorizer()  # 默认使用sparse稀疏矩阵：节省空间、提高加载效率，使用one-hot编码：为了避免类别相对信息大小的产生
    # transfer = DictVectorizer(sparse=False)  # 显示为二维矩阵

    new_data = transfer.fit_transform(data)
    print('特征名称:\n', transfer.get_feature_names())
    print('特征值:\n', new_data.toarray())  # toarray()方法，表示将sparse稀疏矩阵转换为二维矩阵

    return None


def english_text_count_demo():
    """
    英文文本特征提取
    :return:
    """
    data = ["life is short,i like like like python", "life is too long,i dislike python"]
    transfer = CountVectorizer()
    new_data = transfer.fit_transform(data)
    print('特征名称:\n', transfer.get_feature_names())
    print('特征值:\n', new_data.toarray())

    return None


def cut_text(text):
    """
    中文分词
    :param text:
    :return:
    """
    return ' '.join(list(jieba.cut(text)))


def chinese_text_count_demo():
    """
    中文文本特征提取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    cut_list = list()
    for item in data:
        cut_list.append(cut_text(item))

    transfer = CountVectorizer()
    new_data = transfer.fit_transform(cut_list)
    print('特征名称:\n', transfer.get_feature_names())
    print('特征值:\n', new_data.toarray())
    return None


def chinese_text_tfidf_demo():
    """
    中文文本特征提取-tfidf
    理解：
        tfidf = tf * idf   反映了该词在整个语料库中所占重要程度
        tf：term frequency 词频：某一个给定的词语在该文件中出现的频率
        idf：inverse document frequency 逆文档词频：由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到

    例如：
        "非常"，"经济"

        语料库中1000篇文章
        100篇中出现了"非常"
        10篇中出现了"经济"

        文章A 100个词语 10个"非常"
            tf = 10 / 100 = 0.1
            idf = lg 1000 / 100 = 1
            tfidf = 0.1
        文章B 100个词语 10个"经济"
            tf = 10 / 100 = 0.1
            idf = lg 1000 / 10 = 2
            tfidf = 0.2
        对数：
            2^3 = 8
            log 2 8 = 3
            log 10 1000 = 3
            lg 1000 = 3
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    cut_list = list()
    for item in data:
        cut_list.append(cut_text(item))

    transfer = TfidfVectorizer(stop_words=['一种'])
    new_data = transfer.fit_transform(cut_list)
    print('特征名称:\n', transfer.get_feature_names())
    print('特征值:\n', new_data.toarray())

    return None


if __name__ == '__main__':
    # 1、字典特征提取
    # dict_demo()

    # 2、英文文本特征提取
    # english_text_count_demo()

    # 3、中文分词
    # cut_text('在学习上，走上了一条不归路')

    # 4、中文文本特征提取
    # chinese_text_count_demo()

    # 5、中文文本tfidf特征提取
    chinese_text_tfidf_demo()
