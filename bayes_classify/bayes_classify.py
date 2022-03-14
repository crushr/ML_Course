class NBClassify(object):
  def __init__(self, fillNa = 1):
    self.fillNa = 1
    pass

  def train(self, trainSet):
    '''计算先验概率'''
    dictTag = {} # key:类别，value：频数 
    for subTuple in trainSet:
      dictTag[str(subTuple[1])] = 1 if str(subTuple[1]) not in dictTag.keys() else dictTag[str(subTuple[1])] + 1
    # dictTag = {'哺乳动物': 7, '非哺乳动物': 13}
    tagProbablity = {}
    totalFreq = sum([value for value in dictTag.values()]) # 总数
    for key, value in dictTag.items():
      tagProbablity[key] = value / totalFreq
    self.tagProbablity = tagProbablity  # 先验概率，两类的经验概率
    # tagProbablity = {'哺乳动物': 0.35, '非哺乳动物': 0.65}

    '''计算条件概率'''
    dictFeaturesBase = {}
    for subTuple in trainSet:
      for key, value in subTuple[0].items():
        if key not in dictFeaturesBase.keys():
          dictFeaturesBase[key] = {value:1}
        else:
          if value not in dictFeaturesBase[key].keys():
            dictFeaturesBase[key][value] = 1
          else:
            dictFeaturesBase[key][value] += 1
    # dictFeaturesBase = {'胎生': {'是': 7, '否': 13}, '会飞': {'否': 16, '是': 4}, '水中生活': {'否': 11, '是': 5, '有时': 4}, '有腿': {'是': 14, '否': 6}}
    dictFeatures = {}.fromkeys([key for key in dictTag]) # {'哺乳动物': None, '非哺乳动物': None}
    for key in dictFeatures.keys():
      dictFeatures[key] = {}.fromkeys([key for key in dictFeaturesBase])
    # {'哺乳动物': {'胎生': None, '会飞': None, '水中生活': None, '有腿': None}, '非哺乳动物': {'胎生': None, '会飞': None, '水中生活': None, '有腿': None}}
    for key, value in dictFeatures.items():
      for subkey in value.keys():
        value[subkey] = {}.fromkeys([x for x in dictFeaturesBase[subkey].keys()])
    # dictFeatures = {'哺乳动物': {'胎生': {'是': None, '否': None}, '会飞': {'否': None, '是': None}, '水中生活': {'否': None, '是': None, '有时': None}, '有腿': {'是': None, '否': None}}, 
    # '非哺乳动物': {'胎生': {'是': None, '否': None}, '会飞': {'否': None, '是': None}, '水中生活': {'否': None, '是': None, '有时': None}, '有腿': {'是': None, '否': None}}}
    
    # 填充数字
    for subTuple in trainSet:
      for key, value in subTuple[0].items():
        dictFeatures[subTuple[1]][key][value] = 1 if dictFeatures[subTuple[1]][key][value] == None else dictFeatures[subTuple[1]][key][value] + 1

    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        for featureKey, featureValues in fetureValueDict.items():
          if featureValues == None:
            fetureValueDict[featureKey] = 1
    
    # 计算概率
    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        totalCount = sum([x for x in fetureValueDict.values() if x != None])
        for featureKey, featureValues in fetureValueDict.items():
          fetureValueDict[featureKey] = featureValues/totalCount if featureValues != None else None
    # {'哺乳动物': {'胎生': {'是': 0.8571428571428571, '否': 0.14285714285714285}, '会飞': {'否': 0.8571428571428571, '是': 0.14285714285714285}, '水中生活': {'否': 0.625, '是': 0.25, '有时': 0.125}, '有腿': {'是': 0.7142857142857143, '否': 0.2857142857142857}}, 
    # '非哺乳动物': {'胎生': {'是': 0.07692307692307693, '否': 0.9230769230769231}, '会飞': {'否': 0.7692307692307693, '是': 0.23076923076923078}, '水中生活': {'否': 0.46153846153846156, '是': 0.23076923076923078, '有时': 0.3076923076923077}, '有腿': {'是': 0.6923076923076923, '否': 0.3076923076923077}}}
    self.featuresProbablity = dictFeatures

  def classify(self, featureDict):
    resultDict = {}
    for key, value in self.tagProbablity.items():
      iNumList = [] # 哺乳：[0.8571428571428571, 0.8571428571428571, 0.25, 0.2857142857142857]
      for f, v in featureDict.items():
        if self.featuresProbablity[key][f][v]:
          iNumList.append(self.featuresProbablity[key][f][v])
      conditionPr = 1
      for iNum in iNumList:
        conditionPr *= iNum # 所有概率连乘
      resultDict[key] = value * conditionPr # 再乘上先验概率
    resultList = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
    # 所以最终比较的是先验概率*类条件概率密度的大小（各类中样本的分布概率）
    return resultList[0][0]

if __name__ == '__main__':
  trainSet = [
    ({"胎生":"是", "会飞":"否","水中生活":"否","有腿":"是"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"否","有腿":"否"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"是","有腿":"否"}, "非哺乳动物"),
    ({"胎生":"是", "会飞":"否","水中生活":"是","有腿":"否"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"有时","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"否","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"是", "会飞":"是","水中生活":"否","有腿":"是"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"是","水中生活":"否","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"是", "会飞":"否","水中生活":"否","有腿":"是"}, "哺乳动物"),
    ({"胎生":"是", "会飞":"否","水中生活":"是","有腿":"否"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"有时","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"有时","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"是", "会飞":"否","水中生活":"否","有腿":"是"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"是","有腿":"否"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"有时","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"否","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"否", "会飞":"否","水中生活":"否","有腿":"是"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"是","水中生活":"否","有腿":"是"}, "非哺乳动物"),
    ({"胎生":"是", "会飞":"否","水中生活":"是","有腿":"否"}, "哺乳动物"),
    ({"胎生":"否", "会飞":"是","水中生活":"否","有腿":"是"}, "非哺乳动物"),
  ]
  monitor = NBClassify()
  monitor.train(trainSet)
  result = monitor.classify({"胎生":"是", "会飞":"否","水中生活":"是","有腿":"否"})
  print('预测类别:',result)