from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def trainSVM(dataX, dataY, kernelFunction='linear'):
    svm = SVC(kernel=kernelFunction)
    return svm.fit(dataX, dataY)


def testSVM(model, testDataX, testDataY, printValues = False):
    yPred = model.predict(testDataX)
    
    acScore = accuracy_score(testDataY, yPred)
    if printValues: print(acScore)

    return acScore

def trainAndTestSVM(dataX, dataY, kernelFunction='linear', randomState=42, printValues = False,):
    # Random State for comparison reasons
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.25, random_state=randomState)
    model = trainSVM(trainDataX, trainDataY, kernelFunction)

    return testSVM(model, testDataX, testDataY, printValues=printValues)
