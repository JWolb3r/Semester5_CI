from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utilty import MetricsContainer

def trainSVM(dataX, dataY, kernelFunction='rbf'):
    svm = SVC(kernel=kernelFunction)
    return svm.fit(dataX, dataY)


def testSVM(model, testDataX, testDataY, printValues = False):
    yPred = model.predict(testDataX)
    
    f1score=f1_score(y_true=testDataY, y_pred=yPred)

    acScore = accuracy_score(testDataY, yPred)
    if printValues: print(acScore)

    return MetricsContainer(acc=acScore, f1score=f1score)

def trainAndTestSVM(data, features, label, kernelFunction='rbf', randomState=42, printValues = False,):
    dataX = data[features]
    dataY = data[label]

    # Random State for comparison reasons
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.25, random_state=randomState)
    model = trainSVM(trainDataX, trainDataY, kernelFunction)

    return testSVM(model, testDataX, testDataY, printValues=printValues)
