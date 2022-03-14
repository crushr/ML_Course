'''
    一次决策
'''

from math import log

# 创建数据集
def createDataSet():
    dataSet=[[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]

    labels=['年龄','有工作','有自己的房子','信贷情况']
    return dataSet,labels

# 计算经验熵（香农熵）
def calc_ShannonEntro(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt=0.0
    #计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

# 根据某特征和某特征值划分数据集，生成下一节点的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            featVec.pop(axis)  # pop参数是位置
            # featVec.pop(value) #remove参数是值，这里不能用remove，因为值不是唯一的，会出bug，索引是唯一的。
            retDataSet.append(featVec)
    return retDataSet

# 原作者方法，对接，没有直接pop好理解
def splitDataSet2(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择划分最佳特征
def chooseBestFeature_ToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calc_ShannonEntro(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet2(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calc_ShannonEntro((subDataSet))
    
        #信息增益 = 经验熵 - 条件熵
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__=='__main__':
    dataSet,features=createDataSet()
    print("最优索引特征："+str(chooseBestFeature_ToSplit(dataSet)))