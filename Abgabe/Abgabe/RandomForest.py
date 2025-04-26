from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from utilty import MetricsContainer

def trainRF(dataX, dataY, n_estimators=100, randomState=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=randomState)
    return model.fit(dataX, dataY)


def testRF(model, testDataX, testDataY, uniqueLabelValues: list = None, printValues = False):
    yPred = model.predict(testDataX)

    f1score=f1_score(y_true=testDataY, y_pred=yPred, average='macro')

    acScore = accuracy_score(testDataY, yPred)
    if printValues: print(acScore)

    return MetricsContainer(acc=acScore, f1score=f1score)

def trainAndTestRF(data,features, label, n_estimators=100, randomState=42, uniqueLabelValues: list = None, printValues = False):
    dataY = data[label]
    dataX = data[features]
    # Random State for comparison reasons
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.25, random_state=randomState)
    model = trainRF(trainDataX, trainDataY, n_estimators=n_estimators, randomState=randomState)
    
    return testRF(model, testDataX, testDataY, uniqueLabelValues=uniqueLabelValues, printValues=printValues)