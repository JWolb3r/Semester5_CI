from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from preprocessing import getColumns, preprocessing, removeColumns
from kNN import trainAndTestKNN
from SVM import trainAndTestSVM
from utilty import printAndWriteInFile, combineFiles

class PreprocessedData:
    def __init__(self, data, features, label):
        self.data = data
        self.features = features
        self.label = label

#################### Cardio_Data ####################
def preprocessCardioData():

    printAndWriteFileWithDefinedDestination("Preprocess Cardio Data")

    filepath = "DataSets\\cardio_data_processed.csv"
    label = "cardio"

    normalizeFeatures = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]

    data = preprocessing(filepath, 
                        oneHotEncodingFeatures=["bp_category"], 
                        normalizeFeatures=normalizeFeatures)

    removableColumns = ["id", "age_years", "bp_category_encoded", label]
    features = removeColumns(getColumns(data),  removableColumns)

    bestFeatures = ffs(data, features, label, maxIter=1000)
    bestFeatures = ["age", "ap_hi", "ap_lo", "cholesterol"]

    printAndWriteFileWithDefinedDestination(f"Best features: {bestFeatures}")

    return PreprocessedData(data, bestFeatures, label)
    # printAndWriteFileWithDefinedDestination(trainAndTestMLP(data[bestFeatures], label))

    # evolve(data[bestFeatures], label)


#################### Iris ####################
def preprocessIris():

    printAndWriteFileWithDefinedDestination("Preprocess Iris")

    filepath = "DataSets\\Iris.csv"
    label = 'Species'

    normalizeFeatures = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

    data = preprocessing(
        filepath=filepath, 
        oneHotEncodingFeatures=[label], 
        normalizeFeatures=normalizeFeatures)

    # Get onehotencoded feature
    features = getColumns(data)
    label_columns = [col for col in features if col.startswith(label)]  
    removableColumns = ["id"] + label_columns
    features = removeColumns(features,  removableColumns)

    bestFeatures = ffs(data, features, label_columns , maxIter=1000)

    printAndWriteFileWithDefinedDestination(f"Best features: {bestFeatures}")

    return PreprocessedData(data, bestFeatures, label_columns)


#################### Titanic ####################
def preprocessTitanic():
    printAndWriteFileWithDefinedDestination("Preprocess Titanic Data")
    data = "DataSets\\titanic_combined.csv"
    label = 'Survived'

    normalizeFeatures = ['Age', 'Fare']  
    
    data = preprocessing(
        filepath=data, 
        oneHotEncodingFeatures=['Sex', 'Embarked'],  
        normalizeFeatures=normalizeFeatures  
    )

    removableColumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']

    features = removeColumns(getColumns(data), removableColumns)
    data = data[features]
    bestFeatures = ffs(data, features, label, maxIter=1000)

    printAndWriteFileWithDefinedDestination(f"Best Features for Titanic: {bestFeatures}")
    return PreprocessedData(data, bestFeatures, label)


#################### FetalHealth ####################
def preprocessFetalHealth():
    printAndWriteFileWithDefinedDestination("Preprocess Fetal Health Data")
    filepath = "DataSets\\fetal_health.csv"
    label = 'fetal_health'

    normalizeFeatures = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
        'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
        'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
        'histogram_variance', 'histogram_tendency'
    ]
    
    data = preprocessing(
        filepath=filepath,   
        normalizeFeatures=normalizeFeatures,
    )
    
    bestFeatures = ffs(data, normalizeFeatures, label, maxIter=1000)

    printAndWriteFileWithDefinedDestination(f"Best Features for Fetal Health: {bestFeatures}")
    return PreprocessedData(data, bestFeatures, label)



def startKNNAverageCreation(data, features, label, trainRange=10, neighborsRange=10, dataSetName=None):
    printAndWriteFileWithDefinedDestination(f"Start {dataSetName} knn")

    # uniform = neighbors have same weight, distance = neighbors got calculated distance
    sumAccUniform = 0
    sumAccDistance = 0
    bestKnnComb = ['',0]

    # Cross-validation
    for neighbors in range(1, neighborsRange):
        printAndWriteFileWithDefinedDestination(f"Current neighbors: {neighbors}")

        sumAccUniform = 0
        sumAccDistance = 0

        for i in range(trainRange):
            sumAccUniform += trainAndTestKNN(data[features], data[label], neighbors=neighbors, randomState=42+i, knnWeight='uniform')
            sumAccDistance += trainAndTestKNN(data[features], data[label], neighbors=neighbors, knnWeight='distance')

        averageUniformAcc = sumAccUniform / trainRange
        averageDistanceAcc = sumAccDistance / trainRange
        printAndWriteFileWithDefinedDestination(f"Average Acc Uniform: {averageUniformAcc}")
        printAndWriteFileWithDefinedDestination(f"Average Acc Distance: {averageDistanceAcc}")

        if bestKnnComb[1] < sumAccUniform:
            bestKnnComb = ['Uniform', averageUniformAcc]
        elif bestKnnComb[1] < sumAccDistance:
            bestKnnComb = ['Distance', averageDistanceAcc]

    printAndWriteFileWithDefinedDestination(f"Best neighbors: {bestKnnComb}")


def startSVMAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    printAndWriteFileWithDefinedDestination(f"Start {datasetName} SVM")

    bestSvmComb = ['', 0] 

    kernelFunctions = ['linear', 'rbf', 'poly', 'sigmoid']

    # Cross-validation
    for kernel in kernelFunctions:
        printAndWriteFileWithDefinedDestination(f"Current kernel: {kernel}")

        sumAcc = 0

        for i in range(trainRange):
            sumAcc += trainAndTestSVM(data[features], data[label], kernelFunction=kernel)

        averageAcc = sumAcc / trainRange

        printAndWriteFileWithDefinedDestination(f"Average Acc for {kernel}: {averageAcc}")

        if averageAcc > bestSvmComb[1]:
            bestSvmComb = [kernel, averageAcc]

    printAndWriteFileWithDefinedDestination(f"Best SVM kernel: {bestSvmComb}")


def startNNAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    printAndWriteFileWithDefinedDestination(f"Start {datasetName} default-NN")

    # Cross-validation
    sumAcc = 0

    for i in range(trainRange):
        sumAcc += trainAndTestMLP(data[features], data[label])

    averageAcc = sumAcc / trainRange

    printAndWriteFileWithDefinedDestination(f"Average Acc: {averageAcc}")


def printAndWriteFileWithDefinedDestination(content):
    printAndWriteInFile(content, "Logs//BestFeatures.txt")


if __name__ == "__main__":
    printAndWriteFileWithDefinedDestination(preprocessFetalHealth())
    printAndWriteFileWithDefinedDestination(preprocessFetalHealth())
    printAndWriteFileWithDefinedDestination(preprocessTitanic())
    printAndWriteFileWithDefinedDestination(preprocessTitanic())
    printAndWriteFileWithDefinedDestination(preprocessCardioData())
    printAndWriteFileWithDefinedDestination(preprocessCardioData())
    printAndWriteFileWithDefinedDestination(preprocessIris())
    printAndWriteFileWithDefinedDestination(preprocessIris())

