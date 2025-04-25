from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from utilty import MetricsContainer

def trainMLP(dataX, dataY, hiddenLayerSizes, maxIter=4000, randomState=42, activation="relu"):

    model = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation=activation, max_iter=maxIter, random_state=randomState)
    return model.fit(dataX, dataY)


def testMLP(model, testDataX, testDataY, uniqueLabelValues: list = None, printValues = False):
    yPred = model.predict(testDataX)

    f1score=f1_score(y_true=testDataY, y_pred=yPred, average='macro')

    if uniqueLabelValues:
        report = classification_report(testDataY, yPred, output_dict=True)
        if printValues: print(report)

        sum = 0
        for value in uniqueLabelValues:
            sum += report[f"{value}"]["f1-score"]
        return sum / len(uniqueLabelValues)
    else:
        acScore = accuracy_score(testDataY, yPred)
        if printValues: print(acScore)

        return MetricsContainer(acc=acScore, f1score=f1score)

def trainAndTestMLP(data, features, label, maxIter=4000, hiddenLayerSizes=(100), randomState=42, uniqueLabelValues: list = None, printValues = False, activationFunction="relu"):
    dataX = data[features]
    dataY = data[label]
    # Random State for comparison reasons
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.25, random_state=randomState)
    model = trainMLP(trainDataX, trainDataY, hiddenLayerSizes, maxIter=maxIter, randomState=randomState, activation=activationFunction)

    return testMLP(model, testDataX, testDataY, uniqueLabelValues=uniqueLabelValues, printValues=printValues)