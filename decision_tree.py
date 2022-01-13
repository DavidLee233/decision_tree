from sklearn.model_selection import train_test_split # 切分数据集
from sklearn import datasets # 导入数据集
from utile import *

# 加载sklearn包中的乳腺癌患者数据集
data = datasets.load_breast_cancer()
dataset = Bunch2dataframe(data)  # 转换成dataframe类型进行处理，最后一列为标签列
# 共有569行数据，即569个病例，每个病例有30个特征可以看到。第31列数据为标签，其中1代表恶性，0代表良性
# 划分训练集和测试集
# 定义训练集占70%，测试集占30%
train, test = train_test_split(dataset,test_size=0.3)
feature_names = dataset.columns[:-1]        # 获取特征名

# 训练决策树
clf_list, i, test_data, test_target =  best_depth_tree(train, test)      # 训练模型
print("max_depth: " + str(i+1))
clf = clf_list[i]     # 选取测试集预测准确率最大值的模型
y_hat = clf.predict(test_data)
test_right = np.count_nonzero(y_hat == test_target)  # 统计预测正确的个数
print('预测正确数目：', test_right)
print('准确率: %.2f%%' % (100 * float(test_right) / float(len(test_target))))
Draw_tree(clf, "breast_cancer", feature_names=feature_names)     # 绘制决策树




