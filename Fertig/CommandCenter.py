from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from preprocessing import getColumns, preprocessing, removeColumns, deleteNaNValues, printLengthAndColumns
from kNN import trainAndTestKNN
from SVM import trainAndTestSVM
from utilty import printAndWriteInFile
import pandas as pd

class PreprocessedData:
    def __init__(self, data: pd.DataFrame, feature: list, label: str):
        self.data = data
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

    return PreprocessedData(data, features, label)


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

    return PreprocessedData(data, features, label)


#################### Titanic ####################
def preprocessTitanic():
    print("Preprocess Titanic Data")
    data = "DataSets\\titanic_combined.csv"
    label = 'Survived'

    normalizeFeatures = ['Age', 'Fare']  
    oneHotEncodingFeatures = ['Sex', 'Embarked']
    
    data = preprocessing(
        filepath=data, 
        oneHotEncodingFeatures=oneHotEncodingFeatures,  
        normalizeFeatures=normalizeFeatures,
        bPrintInfo=False  
    )

    removableColumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    necessaryValues = removeColumns(getColumns(data), removableColumns)
    data = data[necessaryValues]

    necessaryValues.remove(label)

    # Manuell Nan value removal, because titanic dataset would lose a lot of columns before unecessary columns are removed
    data = deleteNaNValues(data)

    printLengthAndColumns(data)

    return PreprocessedData(data, necessaryValues, label)


#################### FetalHealth ####################
def preprocessFetalHealth():
    print("Preprocess Fetal Health Data")
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
    
    return PreprocessedData(data, normalizeFeatures, label)



def startKNNAverageCreation(data, features, label, trainRange=10, neighborsRange=10, dataSetName=None):
    printAndWriteInFileAcc(f"Start {dataSetName} knn")
    printAndWriteInFileAvgAcc(f"Start {dataSetName} knn")

    # uniform = neighbors have same weight, distance = neighbors got calculated distance
    sumAccUniform = 0
    sumAccDistance = 0
    bestKnnComb = ['','',0]

    # Cross-validation
    for neighbors in range(1, neighborsRange):
        printAndWriteInFileAcc(f"Current neighbors: {neighbors}")
        printAndWriteInFileAvgAcc(f"Current neighbors: {neighbors}")

        sumAccUniform = 0
        sumAccDistance = 0

        printAndWriteInFileAvgAcc("Distance:")
        printAndWriteInFileAcc("Distance")


        for i in range(trainRange):
            distanceAcc = trainAndTestKNN(data=data, features=features, label=label, neighbors=neighbors, randomState=42+i, knnWeight='distance')

            printAndWriteInFileAcc(distanceAcc)

            sumAccDistance += distanceAcc

        averageDistanceAcc = sumAccDistance / trainRange
        printAndWriteInFileAvgAcc(f"Average Acc Distance: {averageDistanceAcc}")


        printAndWriteInFileAcc("Uniform:")
        printAndWriteInFileAvgAcc("Uniform")
        for i in range(trainRange):
            uniformAcc = trainAndTestKNN(data=data, features=features, label=label, neighbors=neighbors, randomState=42+i, knnWeight='uniform')

            printAndWriteInFileAcc(uniformAcc)

            sumAccUniform += uniformAcc

        averageUniformAcc = sumAccUniform / trainRange
        printAndWriteInFileAvgAcc(f"Average Acc Uniform: {averageUniformAcc}")

        if bestKnnComb[2] < averageUniformAcc:
            bestKnnComb = ['Uniform', neighbors , averageUniformAcc]
        elif bestKnnComb[2] < averageDistanceAcc:
            bestKnnComb = ['Distance', neighbors, averageDistanceAcc]

    printAndWriteInFileAcc(f"Best neighbors: {bestKnnComb}")
    printAndWriteInFileAvgAcc(f"Best neighbors: {bestKnnComb}")



def startSVMAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    printAndWriteInFileAcc(f"Start {datasetName} SVM")
    printAndWriteInFileAvgAcc(f"Start {datasetName} SVM")

    bestSvmComb = ['', 0] 
    kernelFunctions = ['linear', 'rbf', 'poly', 'sigmoid']

    # Cross-validation
    for kernel in kernelFunctions:
        printAndWriteInFileAcc(f"Current kernel: {kernel}")
        printAndWriteInFileAvgAcc(f"Current kernel: {kernel}")

        sumAcc = 0

        for i in range(trainRange):
            acc = trainAndTestSVM(data=data, features=features, label=label, kernelFunction=kernel)
            printAndWriteInFileAcc(acc)
            sumAcc += acc

        averageAcc = sumAcc / trainRange
        printAndWriteInFileAvgAcc(f"Average Acc for {kernel}: {averageAcc}")

        if averageAcc > bestSvmComb[1]:
            bestSvmComb = [kernel, averageAcc]

    printAndWriteInFileAcc(f"Best SVM kernel: {bestSvmComb}")
    printAndWriteInFileAvgAcc(f"Best SVM kernel: {bestSvmComb}")



def startNNAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    printAndWriteInFileAcc(f"Start {datasetName} default-NN")
    printAndWriteInFileAvgAcc(f"Start {datasetName} default-NN")

    # Cross-validation
    sumAcc = 0

    for i in range(trainRange):
        acc = trainAndTestMLP(data=data, features=features, label=label)
        printAndWriteInFileAcc(acc)
        sumAcc += acc

    averageAcc = sumAcc / trainRange
    printAndWriteInFileAvgAcc(f"Average Acc: {averageAcc}")


def printAndWriteInFileBestFeatures(content):
    printAndWriteInFile(content, "Logs/BestFeatures.txt")

def printAndWriteInFileAcc(content):
    printAndWriteInFile(content, "Logs/AccssOfAlgorithms.txt")

def printAndWriteInFileAvgAcc(content):
    printAndWriteInFile(content, "Logs/AvgAcssOfAlgs.txt")

def doFFS(datasetName, data, features, label):
    bestFeatures = ffs(datasetName, data, features, label, maxIter=1000)

    print(f"Best Features for Titanic: {bestFeatures}")

    return bestFeatures

    
def titanic_cardio_iris__fetal_analysis(bKNN=False, bNN=False, bSVM=False, bFFS=False):
    datasets = [
        {
            "name": "Titanic",
            "preprocess": preprocessTitanic,
            "features": ['Age', 'Parch', 'Sex_female', 'Sex_male']
        },
        {
            "name": "Cardio",
            "preprocess": preprocessCardioData,
            "features": ['age', 'ap_hi', 'ap_lo', 'cholesterol']
        },
        {
            "name": "Iris",
            "preprocess": preprocessIris,
            "features": None
        },
        {
            "name": "FetalHealth",
            "preprocess": preprocessFetalHealth,
            "features": ['severe_decelerations', 'prolongued_decelerations', 'mean_value_of_short_term_variability', 'histogram_median']
        }
    ]

    for dataset in datasets:
        printAndWriteInFileBestFeatures(f"#######{dataset['name']}#######")
        preprocessed = dataset["preprocess"]()
        features = dataset["features"] or preprocessed.feature
        label = preprocessed.label

        if bFFS:
            printAndWriteInFileBestFeatures(doFFS(dataset["name"], preprocessed.data, preprocessed.feature, label))
        if bKNN:
            startKNNAverageCreation(data=preprocessed.data, features=features, label=label, dataSetName=dataset["name"], neighborsRange=100, trainRange=20)
        if bNN:
            startNNAverageCreation(data=preprocessed.data, features=features, label=label, datasetName=dataset["name"], trainRange=20)
        if bSVM:
            startSVMAverageCreation(data=preprocessed.data, features=features, label=label, datasetName=dataset["name"], trainRange=20)

    
    
    
    

if __name__ == "__main__":
    titanic_cardio_iris__fetal_analysis(bKNN=True, bNN=True, bSVM=True)