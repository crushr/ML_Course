class NBClassify(object):
  def __init__(self, fillNa = 1):
    self.fillNa = 1
    pass
  def train(self, trainSet):
    dictTag = {}
    for subTuple in trainSet:
      dictTag[str(subTuple[1])] = 1 if str(subTuple[1]) not in dictTag.keys() else dictTag[str(subTuple[1])] + 1
    tagProbablity = {}
    totalFreq = sum([value for value in dictTag.values()])
    for key, value in dictTag.items():
      tagProbablity[key] = value / totalFreq
    self.tagProbablity = tagProbablity
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

    dictFeatures = {}.fromkeys([key for key in dictTag])
    for key in dictFeatures.keys():
      dictFeatures[key] = {}.fromkeys([key for key in dictFeaturesBase])
    for key, value in dictFeatures.items():
      for subkey in value.keys():
        value[subkey] = {}.fromkeys([x for x in dictFeaturesBase[subkey].keys()])

    for subTuple in trainSet:
      for key, value in subTuple[0].items():
        dictFeatures[subTuple[1]][key][value] = 1 if dictFeatures[subTuple[1]][key][value] == None else dictFeatures[subTuple[1]][key][value] + 1

    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        for featureKey, featureValues in fetureValueDict.items():
          if featureValues == None:
            fetureValueDict[featureKey] = 1

    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        totalCount = sum([x for x in fetureValueDict.values() if x != None])
        for featureKey, featureValues in fetureValueDict.items():
          fetureValueDict[featureKey] = featureValues/totalCount if featureValues != None else None
    self.featuresProbablity = dictFeatures

  def classify(self, featureDict):
    resultDict = {}
    for key, value in self.tagProbablity.items():
      iNumList = []
      for f, v in featureDict.items():
        if self.featuresProbablity[key][f][v]:
          iNumList.append(self.featuresProbablity[key][f][v])
      conditionPr = 1
      for iNum in iNumList:
        conditionPr *= iNum
      resultDict[key] = value * conditionPr
    resultList = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
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