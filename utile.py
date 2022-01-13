import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn import tree	# 导入树
import graphviz
import numpy as np

# 将sklearn数据集Bunch类型转成DataFrame
def Bunch2dataframe(sklearn_dataset):
    """
    将sklearn数据集Bunch类型转成DataFrame
    :param sklearn_dataset: sklearn中的数据集
    :return: 处理后的dataframe，最后一列为标签列
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)        # 追加一列标签列
    return df

def best_depth_tree(train, test):
    """
    调参得到最佳的max_depth值并返回对应训练后的模型
    :param train: 训练集
    :param test: 测试集
    :return: 训练后的模型列表和测试集预测准确率最大值的索引
    """
    train_score_list = []
    test_score_list = []
    clf_list = []
    max_test_depth = 4     # 最大树深(超参数上限)
    train_data = train.iloc[:, :-1] # 去除所有行中的最后一列的剩余部分
    train_target = train.iloc[:, -1] # 所有行中的最后一列
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    for i in range(max_test_depth):
        clf = DecisionTreeClassifier(criterion="gini", max_depth=i+1, random_state=30, splitter="random")
        clf = clf.fit(train_data, train_target)     # 训练模型
        score_train = clf.score(train_data, train_target)       # 训练集预测准确率
        score_test = clf.score(test_data, test_target)       # 测试集预测准确率
        train_score_list.append(score_train)
        test_score_list.append(score_test)
        clf_list.append(clf)
    plt.plot(range(1, max_test_depth+1), train_score_list, color="blue", label="train")        # 绘制分数曲线
    plt.plot(range(1, max_test_depth+1), test_score_list, color="red", label="test")
    plt.xlabel('决策树深度', fontproperties="SimSun")
    plt.ylabel('准确率', fontproperties="SimSun")
    plt.title('训练集和测试集预测准确率分布', fontproperties="SimSun")
    plt.legend()
    plt.show()
    return clf_list, test_score_list.index(max(test_score_list)), test_data, test_target

def Draw_tree(clf, filename, feature_names=None, class_names=None):
    """
    绘制决策树并保存为*.pdf文件
    :param clf: 训练后的模型
    :param filename: 保存的文件名
    :param feature_names: 特征名
    :param class_names: 标签名
    :return: None
    """
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename)
    print("Done.")
