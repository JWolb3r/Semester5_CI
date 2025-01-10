from RandomForest import trainAndTestRF
from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from Preprocessing import getColumns, preprocessing, removeColumns, deleteNaNValues, printLengthAndColumns
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

    # Good preprocessing

    normalizeFeatures = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]
    data = preprocessing(filepath, 
                        oneHotEncodingFeatures=["bp_category"], 
                        normalizeFeatures=normalizeFeatures,
                        bDeleteNanValues=True              
                        )
    removableColumns = ["id", "age_years", "bp_category_encoded", label]

    # Bad preprocessing
    # data = preprocessing(filepath, 
    #                     oneHotEncodingFeatures=["bp_category"],              
    #                     )
    # removableColumns = ["age_years", "bp_category_encoded",label]

    # Necessary operation
    features = removeColumns(getColumns(data),  removableColumns)

    return PreprocessedData(data, features, label)


#################### Iris ####################
def preprocessIris():

    print("\nPreprocess Iris")

    filepath = "DataSets\\Iris.csv"
    label = 'Species'

    # Good preprocessing
    normalizeFeatures = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    data = preprocessing(
        filepath=filepath, 
        labelEncodingFeatures=[label], 
        normalizeFeatures=normalizeFeatures,
        bDeleteNanValues=True
        )
    removableColumns = ["Id"] + [label]

    # For onehotencoded label
    # features = getColumns(data)
    # label = [col for col in features if col.startswith(label)]  

    # Bad preprocessing
    # data = preprocessing(
    #     filepath=filepath, 
    #     labelEncodingFeatures=[label]
    #     )
    # removableColumns = [label]


    # Necessary operation: remove columns manuel, because we just know all labels after onehotencoding species label
    features = getColumns(data)
    features = removeColumns(features,  removableColumns)

    return PreprocessedData(data, features, label)


#################### Titanic ####################
def preprocessTitanic():
    print("Preprocess Titanic Data")
    data = "DataSets\\titanic_combined.csv"
    label = 'Survived'

    # Good preprocessing
    normalizeFeatures = ['Age', 'Fare']  
    oneHotEncodingFeatures = ['Sex', 'Embarked']
    
    data = preprocessing(
        filepath=data, 
        oneHotEncodingFeatures=oneHotEncodingFeatures,  
        normalizeFeatures=normalizeFeatures,
        bPrintInfo=False  
    )
    removableColumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']

    # Bad preprocessing
    # oneHotEncodingFeatures = ['Sex', 'Embarked']
    # data = preprocessing(
    #     filepath=data, 
    #     oneHotEncodingFeatures=oneHotEncodingFeatures,  
    #     bPrintInfo=False  
    # )
    # removableColumns = ["Name", "Ticket" ,"Cabin"]


    # Necessary operations:
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

    # Good preprocessing
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

    # Bad preprocessing
    # data = preprocessing(
    #     filepath=filepath
    # )
    # removableColumns = [label]
    # normalizeFeatures = removeColumns(getColumns(data), removableColumns)
    
    
    return PreprocessedData(data, normalizeFeatures, label)

def startRFAvgCreation(data, features, label,trainRange=10, dataSetName=None):
    printAndWriteInFileAcc(f"Start {dataSetName} rf")
    printAndWriteInFileAvgAcc(f"Start {dataSetName} rf")
    sumAcc = 0
    for i in range(trainRange):
        acc = trainAndTestRF(data, features=features, label=label, randomState=42+i)

        printAndWriteInFileAcc(acc)

        sumAcc += acc

    avgAcc = sumAcc / trainRange
    printAndWriteInFileAvgAcc(f"RF Average Acc: {avgAcc}")

def startEA(data, features, label, maxIterationsMLP=1000, maxIterEa=10):
    evolve(data=data,features=features , label=label, maxIterations=maxIterEa, maxIter=maxIterationsMLP)

# EA Test
def fitness(genome: [], *_args, **_kwargs):
    return sum(genome)

def testEA():
    evolve(None,"",maxIterations=2000,popSize=500,fitness=fitness)


def findBestKNNComb(data, features, label, trainRange=10, neighborsRange=10, dataSetName=None ):
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

        for i in range(trainRange):
            distanceAcc = trainAndTestKNN(data=data, features=features, label=label, neighbors=neighbors, randomState=42+i, knnWeight='distance')

            printAndWriteInFileAcc(distanceAcc)

            sumAccDistance += distanceAcc

        averageDistanceAcc = sumAccDistance / trainRange
        printAndWriteInFileAvgAcc(f"Average Acc Distance: {averageDistanceAcc}")

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

    printAndWriteInFileBestComb("KNN")
    printAndWriteInFileBestComb(f"DataSet: {dataSetName} with trainrange: {trainRange} and neighborsrange: {neighborsRange}")
    printAndWriteInFileBestComb(f"Best comb: {bestKnnComb}")

def startKNNAverageCreation(data, features, label,weights, trainRange=10, neighborsRange=10, dataSetName=None):
    printAndWriteInFileAcc(f"Start {dataSetName} knn with neighbors: {neighborsRange} and weights: {weights}")
    printAndWriteInFileAvgAcc(f"Start {dataSetName} knn")
    sumAcc = 0
    for i in range(trainRange):
        acc = trainAndTestKNN(data=data, features=features, label=label, neighbors=neighborsRange, randomState=42+i, knnWeight=weights)

        printAndWriteInFileAcc(acc)

        sumAcc += acc

    avgAcc = sumAcc / trainRange
    printAndWriteInFileAvgAcc(f"Average Acc with neighbors {neighborsRange} and weight {weights}: {avgAcc}")

def findBestSVMComb(data, features, label, trainRange=10, datasetName=None): 
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
            acc = trainAndTestSVM(data=data, features=features, label=label, kernelFunction=kernel, randomState=42+i)
            printAndWriteInFileAcc(acc)
            sumAcc += acc

        averageAcc = sumAcc / trainRange
        printAndWriteInFileAvgAcc(f"Average Acc for {kernel}: {averageAcc}")

        if averageAcc > bestSvmComb[1]:
            bestSvmComb = [kernel, averageAcc]
    printAndWriteInFileBestComb("SVM:")
    printAndWriteInFileBestComb(f"DataSet: {datasetName} with trainrange: {trainRange}")
    printAndWriteInFileBestComb(f"Best  kernel: {bestSvmComb}")

def startSVMAverageCreation(data, features, label,kernel, trainRange=10, datasetName=None): 
    printAndWriteInFileAcc(f"Start {datasetName} SVM with kernel: {kernel}")
    printAndWriteInFileAvgAcc(f"Start {datasetName} SVM")

    sumAcc = 0
    for i in range(trainRange):
        acc = trainAndTestSVM(data=data, features=features, label=label, kernelFunction=kernel, randomState=42+i)
        printAndWriteInFileAcc(acc)
        sumAcc += acc

    averageAcc = sumAcc / trainRange
    printAndWriteInFileAvgAcc(f"Average Acc for {kernel}: {averageAcc}")


def startNNAverageCreation(data, features, label, trainRange=10, datasetName=None): 
    printAndWriteInFileAcc(f"Start {datasetName} default-NN")
    printAndWriteInFileAvgAcc(f"Start {datasetName} default-NN")
    # Cross-validation
    sumAcc = 0

    for i in range(trainRange):
        acc = trainAndTestMLP(data=data, features=features, label=label, randomState=42+i)
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

def printAndWriteInFileBestComb(content):
    printAndWriteInFile(content, "Logs/BestComb.txt")

def doFFS(datasetName, data, features, label):
    bestFeatures = ffs(data, features, label, maxIter=1000)

    printAndWriteInFileBestFeatures(f"Best Features for {datasetName}: {bestFeatures}")

    return bestFeatures

    
def titanic_cardio_iris__fetal_analysis(bEA=False, bRF=False, bKNN=False, bNN=False, bSVM=False, bFFS=False, bfindComb = False, bCreateAccs = False, trainRange = 20):
    """
    This method performs preprocessing, analysis, and evaluation on four datasets: Titanic, Cardio, Iris, and FetalHealth.
    It supports various knn, svm, rf, default-nn, including feature selection, evolutionary algorithms and accuracy 
    computations.

    Parameters:
    - bEA (bool, default=False): If True, runs an evolutionary algorithm (EA) for the nn layer/neuron optimization. Results are printed 
      to the console only.
    - bRF (bool, default=False): If True, computes average accuracies using Random Forest and logs the results to a text file 
      in the 'logs' directory.
    - bKNN (bool, default=False): If True, performs K-Nearest Neighbors (KNN) experiments (best combination search and 
      average accuracy computation). Outputs are logged to a text file.
    - bNN (bool, default=False): If True, computes average accuracies using a default Neural Networks and logs the results to a text file.
    - bSVM (bool, default=False): If True, runs Support Vector Machine (SVM) experiments (best combination search and 
      average accuracy computation). Outputs are logged to a text file.
    - bFFS (bool, default=False): If True, performs forward feature selection (FFS). Note: FFS is skipped for the Iris dataset 
      because it only has 4 features and the ffs searches per default 4 best features.
    - bfindComb (bool, default=False): If True, searches for the best knn and svm combinations. Results are printed in 
      the console and logged to a text file.
    - bCreateAccs (bool, default=False): If True, computes average accuracies for various methods (KNN, SVM, NN, RF). Results 
      are logged to text files.
    - trainRange (int, default=20): Specifies the number of accuracy measurements used to calculate the average accuracy during model evaluation. 

    Workflow:
    1. Preprocesses the dataset using the appropriate preprocessing function.
    2. Prints dataset information (features, labels, and length) to the console.
    3. If `bFFS` is True, performs forward feature selection unless the dataset is Iris.
    4. If `bEA` is True, runs the evolutionary algorithm and prints the results to the console only.
    5. If `bfindComb` is True, finds the best svm and knn combinations and logs the results to text files.
    6. If `bCreateAccs` is True, computes average accuracies for KNN, SVM, NN, and RF and logs the results to text files.

    Notes:
    - All methods, except for the evolutionary algorithm (`bEA`), log their results to text files located in the 'logs' 
      directory. Some outputs may appear multiple times in the console if they are also written to the logs.
    - Forward feature selection is not applicable to the Iris dataset because it only has 4 features, which matches the default 
      number of features selected.
    """
    datasets = [
        {
            "name": "Titanic",
            "preprocess": preprocessTitanic,
            "features": ['Age', 'Parch', 'Sex_female', 'Sex_male'],
            "knn": {"neighbors": 1,
                    "weights": "uniform"},
            "svmKernel": "linear" 
        },
        {
            "name": "Cardio",
            "preprocess": preprocessCardioData,
            "features": ['age', 'ap_hi', 'ap_lo', 'cholesterol'],
            "knn": {"neighbors": 1,
                    "weights": "uniform"},
            "svmKernel": "linear" 
        },
        {
            "name": "Iris",
            "preprocess": preprocessIris,
            "features": None,
            "knn": {"neighbors": 1,
                    "weights": "uniform"},
            "svmKernel": "linear" 
        },
        {
            "name": "FetalHealth",
            "preprocess": preprocessFetalHealth,
            "features": ['severe_decelerations', 'prolongued_decelerations', 'mean_value_of_short_term_variability', 'histogram_median'],
            "knn": {"neighbors": 1,
                    "weights": "distance"},
            "svmKernel": "linear" 
        }
    ]

    for dataset in datasets:
        preprocessed = dataset["preprocess"]()
        features = dataset["features"] if "features" in dataset and dataset["features"] else preprocessed.feature
        label = preprocessed.label

        print(f"#######{dataset['name']}#######")
        print(f"Dataset Informations")
        print(f"Features: {features}")
        print(f"Label: {label}")
        print(f"Length: {len(preprocessed.data)}")
        

        if bFFS:
            doFFS(dataset["name"], preprocessed.data, preprocessed.feature, label)
        
        if bEA:
            startEA(data=preprocessed.data, features=features, label=label)

        if bfindComb:
            printAndWriteInFileBestComb(f"#######{dataset['name']}#######")
            printAndWriteInFileBestComb(f"Dataset Informations")
            printAndWriteInFileBestComb(f"Features: {features}")
            printAndWriteInFileBestComb(f"Label: {label}")
            printAndWriteInFileBestComb(f"Length: {len(preprocessed.data)}")

            if bKNN:
                findBestKNNComb(data=preprocessed.data, features=features, label=label, dataSetName=dataset["name"], neighborsRange=30, trainRange=trainRange)
            if bSVM:
                findBestSVMComb(data=preprocessed.data, features=features, label=label, datasetName=dataset["name"], trainRange=trainRange)

        if bCreateAccs:

            printAndWriteInFileAvgAcc(f"#######{dataset['name']}#######")
            printAndWriteInFileAvgAcc(f"Dataset Informations")
            printAndWriteInFileAvgAcc(f"Features: {features}")
            printAndWriteInFileAvgAcc(f"Label: {label}")
            printAndWriteInFileAvgAcc(f"Length: {len(preprocessed.data)}")

            if bKNN:
                startKNNAverageCreation(data=preprocessed.data, features=features, label=label, dataSetName=dataset["name"], neighborsRange=dataset["knn"]["neighbors"], weights=dataset["knn"]["weights"], trainRange=trainRange)
            if bSVM:
                startSVMAverageCreation(data=preprocessed.data, features=features, label=label, datasetName=dataset["name"], kernel=dataset["svmKernel"], trainRange=trainRange)
            if bNN:
                startNNAverageCreation(data=preprocessed.data, features=features, label=label, datasetName=dataset["name"], trainRange=trainRange)
            if bRF:
                startRFAvgCreation(data=preprocessed.data, features=features, label=label, trainRange=trainRange, dataSetName=dataset["name"])
            

                

if __name__ == "__main__":
    # Test call
    titanic_cardio_iris__fetal_analysis(bEA=True, trainRange=10)
