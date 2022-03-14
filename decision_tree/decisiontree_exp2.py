'''
    多次决策构建决策树
'''

from decisiontree_exp1 import *
import operator

# 统计classList中出现次数最多的元素（类标签）
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet,labels,featLabels):
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeature_ToSplit(dataSet)  # 得到最优特征的索引bestFeat
    bestFeatLabel = labels[bestFeat]  # 得到最优特征的特征名
    featLabels.append(bestFeatLabel)  # 将最优特征名append进featLabels featLabels不断迭代加入当前最优的特征名
    myTree = {bestFeatLabel:{}}  # 构建myTree字典，最优特征的特征名作为最外层的key
    del(labels[bestFeat]) # 删除labels中的最优特征的特征名
    featValues = [example[bestFeat] for example in dataSet]  # 取出所有的最优特征
    uniqueVls = set(featValues) # 化为唯一的最优特征

    # 遍历唯一值value，把下一个迭代决策树赋给myTree[bestFeatLabel][value]，下一个迭代决策树的dataset是splitDataSet挑选出来的数据
    for value in uniqueVls:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),
                                                  labels,
                                                  featLabels)
    return myTree

if __name__=='__main__':
    dataSet,labels=createDataSet()
    featLabels=[]
    myTree=createTree(dataSet,labels,featLabels)
    print(myTree)