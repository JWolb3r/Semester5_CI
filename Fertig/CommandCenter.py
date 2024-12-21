from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from preprocessing import getColumns, preprocessing, removeColumns, deleteNaNValues, printLengthAndColumns
from kNN import trainAndTestKNN
from SVM import trainAndTestSVM
from utilty import printAndWriteInFile
import pandas as pd

class PreprocessedData:
    def __init__(self, dataX: pd.DataFrame, dataY: pd.DataFrame, feature: list, label: list):
        self.dataX = dataX,
        self.dataY = dataY,
        self.feature = feature
        self.label = label

#################### Cardio_Data ####################
def preprocessCardioData():
    print("Preprocess Cardio Data")

    filepath = "DataSets\\cardio_data_processed.csv"
    label = "cardio"

    normalizeFeatures = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]

    data = preprocessing(filepath, 
                        oneHotEncodingFeatures=["bp_category"], 
                        normalizeFeatures=normalizeFeatures,
                        bDeleteNanValues=True              
                        )


    removableColumns = ["id", "age_years", "bp_category_encoded", label]
    features = removeColumns(getColumns(data),  removableColumns)

    data = data[features+[label]]

    printLengthAndColumns(data)

    return PreprocessedData(data[features], data[label], features, label)
    # print(trainAndTestMLP(data[bestFeatures], label))

    # evolve(data[bestFeatures], label)


#################### Iris ####################
def preprocessIris():

    print("\nPreprocess Iris")

    filepath = "DataSets\\Iris.csv"
    label = 'Species'

    normalizeFeatures = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

    data = preprocessing(
        filepath=filepath, 
        oneHotEncodingFeatures=[label], 
        normalizeFeatures=normalizeFeatures,
        bDeleteNanValues=True)

    # remove columns manuel, because we just know all labels after onehotencoding species label
    features = getColumns(data)
    label = [col for col in features if col.startswith(label)]  
    removableColumns = ["Id"] + label
    features = removeColumns(features,  removableColumns)

    data = data[features + label]

    printLengthAndColumns(data)

    return PreprocessedData(data, data[label], features, label)


#################### Titanic ####################
def preprocessTitanic():
    print("Preprocess Titanic Data")
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
    data = data[features+[label]]

    # Manuell Nan value removal, because titanic dataset would lose a lot of columns before unecessary columns are removed
    data = deleteNaNValues(data)

    printLengthAndColumns(data)

    return PreprocessedData(data[features], data[label], features, label)


#################### FetalHealth ####################
def preprocessFetalHealth():
    print("\nPreprocess Fetal Health Data")
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
        bDeleteNanValues=True
    )
    
    printLengthAndColumns(data)

    return PreprocessedData(data[normalizeFeatures], data[label], normalizeFeatures, label)



def startKNNAverageCreation(data, features, label, trainRange=10, neighborsRange=10, dataSetName=None):
    print(f"Start {dataSetName} knn")

    # uniform = neighbors have same weight, distance = neighbors got calculated distance
    sumAccUniform = 0
    sumAccDistance = 0
    bestKnnComb = ['',0]

    # Cross-validation
    for neighbors in range(1, neighborsRange):
        print(f"Current neighbors: {neighbors}")

        sumAccUniform = 0
        sumAccDistance = 0

        for i in range(trainRange):
            sumAccUniform += trainAndTestKNN(data[features], data[label], neighbors=neighbors, randomState=42+i, knnWeight='uniform')
            sumAccDistance += trainAndTestKNN(data[features], data[label], neighbors=neighbors, knnWeight='distance')

        averageUniformAcc = sumAccUniform / trainRange
        averageDistanceAcc = sumAccDistance / trainRange
        print(f"Average Acc Uniform: {averageUniformAcc}")
        print(f"Average Acc Distance: {averageDistanceAcc}")

        if bestKnnComb[1] < sumAccUniform:
            bestKnnComb = ['Uniform', averageUniformAcc]
        elif bestKnnComb[1] < sumAccDistance:
            bestKnnComb = ['Distance', averageDistanceAcc]

    print(f"Best neighbors: {bestKnnComb}")


def startSVMAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    print(f"Start {datasetName} SVM")

    bestSvmComb = ['', 0] 

    kernelFunctions = ['linear', 'rbf', 'poly', 'sigmoid']

    # Cross-validation
    for kernel in kernelFunctions:
        print(f"Current kernel: {kernel}")

        sumAcc = 0

        for i in range(trainRange):
            sumAcc += trainAndTestSVM(data[features], data[label], kernelFunction=kernel)

        averageAcc = sumAcc / trainRange

        print(f"Average Acc for {kernel}: {averageAcc}")

        if averageAcc > bestSvmComb[1]:
            bestSvmComb = [kernel, averageAcc]

    print(f"Best SVM kernel: {bestSvmComb}")


def startNNAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    print(f"Start {datasetName} default-NN")

    # Cross-validation
    sumAcc = 0

    for i in range(trainRange):
        sumAcc += trainAndTestMLP(data[features], data[label])

    averageAcc = sumAcc / trainRange

    print(f"Average Acc: {averageAcc}")

def doFFS(datsetName, data, features, label):
    bestFeatures = ffs("FetalHealth",data, features, label, maxIter=1000)

    print(f"Best Features for Fetal Health: {bestFeatures}")
    return bestFeatures

def printAndWriteInFileBestFeatures(content):
    printAndWriteInFile(content, "Logs/BestFeatures.txt")

if __name__ == "__main__":
    printAndWriteInFileBestFeatures("\n#######FetalHealth#######")
    fetalHealthpreprocessed = preprocessFetalHealth()
    printAndWriteInFileBestFeatures(doFFS("Fetal",fetalHealthpreprocessed.data, fetalHealthpreprocessed.feature, fetalHealthpreprocessed.label))

    print("\n#######Iris Preprocess#######")
    irisPreprocessed = preprocessIris()

    printAndWriteInFileBestFeatures("\n#######Titanic#######")
    titanicPreprocessed = preprocessTitanic()
    printAndWriteInFileBestFeatures(doFFS("Titanic",titanicPreprocessed.data, titanicPreprocessed.feature, titanicPreprocessed.label))

    printAndWriteInFileBestFeatures("\n#######Cardio#######")
    cardioPreprocessed = preprocessCardioData()
    printAndWriteInFileBestFeatures(doFFS("Cardio",cardioPreprocessed.data, cardioPreprocessed.feature, cardioPreprocessed.label))

