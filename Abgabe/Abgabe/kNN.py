from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utilty import MetricsContainer

def trainKNN(dataX, dataY, neighbors=3, knnWeight="uniform"):
    # Standard euclidean for distance
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights=knnWeight)
    return knn.fit(dataX, dataY)


def testKNN(model, testDataX, testDataY, printValues = False):
    yPred = model.predict(testDataX)
    
    f1score=f1_score(y_true=testDataY, y_pred=yPred)

    acScore = accuracy_score(testDataY, yPred)
    if printValues: print(acScore)

    return MetricsContainer(acc=acScore, f1score=f1score)

def trainAndTestKNN(data, features, label, neighbors=5, randomState=42, printValues = False, knnWeight="uniform"):
    dataX = data[features]
    dataY = data[label]

    # Random State for comparison reasons
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.25, random_state=randomState)
    model = trainKNN(trainDataX, trainDataY, neighbors,knnWeight)

    return testKNN(model, testDataX, testDataY, printValues=printValues)
