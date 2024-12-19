from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from preprocessing import getColumns, preprocessing, removeColumns
from kNN import trainAndTestKNN
from SVM import trainAndTestSVM


#################### Cardio_Data ####################
def preprocessCardioData():
    print("Preprocess Cardio Data")

    cardioFilepath = "DataSets\\cardio_data_processed.csv"
    cardioLabel = "cardio"
    cardioData = preprocessing(cardioFilepath, oneHotEncodingFeatures=["bp_category"], normalizeFeatures=["age", "height", "weight", "ap_hi", "ap_lo", "bmi"])

    columns = removeColumns(getColumns(cardioData), ["id", "age_years", "bp_category_encoded", cardioLabel] )

    bestFeatures = ffs(cardioData, columns, cardioLabel, maxIter=1000)
    bestFeatures = ["age", "ap_hi", "ap_lo", "cholesterol"]
    bestFeatures.append(cardioLabel)

    print(f"Beste features: {bestFeatures}")


    return bestFeatures
    # print(trainAndTestMLP(cardioData[bestFeatures], cardioLabel))

    # evolve(cardioData[bestFeatures], cardioLabel)


#################### Iris ####################
def preprocessIris():
    print("Preprocess Iris")
    irisFilepath = "DataSets\\Iris.csv"
    irisLabel = 'Species'

    irisData = preprocessing(filepath=irisFilepath, oneHotEncoded=[irisLabel], normalizeFeatures=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    label_columns = [col for col in columns.columns if col.startswith(irisLabel)]
    columns = removeColumns(getColumns(irisData), ["id", label_columns] )

    bestFeatures = ffs(irisData, columns, label_columns , maxIter=1000)
    bestFeatures.append(label_columns)

    print(f"Beste features: {bestFeatures}")

    return bestFeatures


#################### Titanic ####################
def preprocessTitanic():
    print("Preprocess Titanic Data")
    titanicData = "DataSets\\titanic_combined.csv"
    titanicLabel = 'Survived'

    normalizeFeatures = ['Age', 'Fare']  
    
    titanicData = preprocessing(
        filepath=titanicData, 
        oneHotEncodingFeatures=['Sex', 'Embarked'],  
        normalizeFeatures=normalizeFeatures  
    )

    columns = removeColumns(getColumns(titanicData), ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    bestFeatures = ffs(titanicData, columns, titanicLabel, maxIter=1000)
    bestFeatures.append(titanicLabel)


    print(f"Beste Features für Titanic: {bestFeatures}")
    return bestFeatures


#################### FetalHealth ####################
def preprocessFetalHealth():
    print("Preprocess Fetal Health Data")
    fetalHealthFilepath = "DataSets\\fetal_health.csv"
    fetalHealthLabel = 'fetal_health'

    normalizeFeatures = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
        'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
        'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
        'histogram_variance', 'histogram_tendency'
    ]
    
    fetalHealthData = preprocessing(
        filepath=fetalHealthFilepath,   
        normalizeFeatures=normalizeFeatures,
    )
    
    bestFeatures = ffs(fetalHealthData, normalizeFeatures, fetalHealthLabel, maxIter=1000)
    bestFeatures.append(fetalHealthLabel)


    print(f"Beste Features für Fetal Health: {bestFeatures}")
    return bestFeatures



def startKNNAverageCreation(data, features, label, trainRange=10, neighborsRange=10, dataSetName=None):
    print(f"Start {dataSetName} knn")

    # uniform = neighbors have same weight, distance = neighbors got calculed distance
    sumAccUniform = 0
    sumAccDistance = 0
    bestKnnComb = ['',0]
    # Cross validation
    for neighbors in range(1,neighborsRange):
        print(f"Current neighbors: {neighbors}")

        sumAccUniform = 0
        sumAccDistance = 0

        for i in range(trainRange):
            sumAccUniform += trainAndTestKNN(data[features], label, neighbors=neighbors, randomState=42+i, knnWeight='uniform')
            sumAccDistance += trainAndTestKNN(data[features], label, neighbors=neighbors, knnWeight='distance')

        averageUniformAcc = sumAccUniform/trainRange
        averageDistanceAcc = sumAccDistance/trainRange
        print(f"Average Ac Uniform: {averageUniformAcc}")
        print(f"Average Ac Distance: {averageDistanceAcc}")

        if bestKnnComb[1] < sumAccUniform:
            bestKnnComb = ['Uniform', averageUniformAcc]
        elif bestKnnComb[1] < sumAccDistance:
            bestKnnComb = ['Distance', averageDistanceAcc]

    print(f"Best neighbors: {bestKnnComb}")


def startSVMAverageCreation(data, features, label,trainRange=10, datasetName=None): 
    # Die Logik hier
    print(f"Start {datasetName} SVM")


    bestSvmComb = ['', 0] 

    kernelFunctions = ['linear', 'rbf', 'poly', 'sigmoid']

    # Cross validation
    for kernel in kernelFunctions:
        print(f"Current kernel: {kernel}")

        sumAccKernel = 0

        for i in range(trainRange):
            sumAccKernel += trainAndTestSVM(data[features], label, kernelFunction=kernel)

        averageKernelAcc = sumAccKernel / trainRange

        print(f"Average Acc for {kernel}: {averageKernelAcc}")

        if averageKernelAcc > bestSvmComb[1]:
            bestSvmComb = [kernel, averageKernelAcc]

    print(f"Best SVM kernel: {bestSvmComb}")

if __name__ == "__main__":
    preprocessCardioData()
    preprocessFetalHealth()
    preprocessIris()
    preprocessTitanic()

