import pandas as pd 
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utilty import printAndWriteInFile
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore


def loadData(filepath):
    return pd.read_csv(filepath)

def removeDuplicates(data):
    data.drop_duplicates(inplace=True)
    return data

def printNullValues(data):
    printAndWriteFileWithDefinedFile("Null Werte Absolut:")
    printAndWriteFileWithDefinedFile(data.isnull().sum())
    printAndWriteFileWithDefinedFile("\nNull Werte Prozentual:")
    printAndWriteFileWithDefinedFile(data.isnull().mean() * 100)
    printAndWriteFileWithDefinedFile("\n")
    return data.dropna(axis=0, how="any")
    
def detectAndRemoveOutliers(data, columns):
    total_outliers = 0
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_count = len(outliers)
        total_outliers += outliers_count
        
        if not outliers.empty:
            largest_outlier = outliers.max()[column]
            smallest_outlier = outliers.min()[column]
        else:
            largest_outlier = None
            smallest_outlier = None
        
        mean_value = data[column].mean()
        
        printAndWriteFileWithDefinedFile(f"Spalte: {column}")
        printAndWriteFileWithDefinedFile(f"Unterer Whisker: {lower_bound}")
        printAndWriteFileWithDefinedFile(f"Oberer Whisker: {upper_bound}")
        printAndWriteFileWithDefinedFile(f"Anzahl der Ausreißer in {column}: {outliers_count}")
        printAndWriteFileWithDefinedFile(f"Größter Ausreißer in {column}: {largest_outlier}")
        printAndWriteFileWithDefinedFile(f"Kleinster Ausreißer in {column}: {smallest_outlier}")
        printAndWriteFileWithDefinedFile(f"Mittelwert von {column}: {mean_value}")
        printAndWriteFileWithDefinedFile("-" * 50)

        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    printAndWriteFileWithDefinedFile(f"Gesamtzahl der Ausreißer über alle Spalten: {total_outliers}")
    return data

def labelEncoded(data,feature):
    encoder = LabelEncoder()
    data[feature] = encoder.fit_transform(data[feature])
    return data

def oneHotEncoded(data, columns: list):
    return pd.get_dummies(data, columns=columns)

def normalizeData(data, selected_features: list):
    scaler = MinMaxScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])
    return data

def preprocessing(
        filepath,
        bRemoveDuplicates = True,
        outlierColumns: list = None,
        labelEncodingFeatures: list = None,
        oneHotEncodingFeatures: list = None,
        normalizeFeatures: list = None,
        bprintNullValues = True,
        bprintData = True,
        bShowBoxplot = False
        ):

    printAndWriteFileWithDefinedFile(f"Start preprocessing of file {filepath}")


    data = loadData(filepath)

    if bRemoveDuplicates:
        data = removeDuplicates(data)

    if outlierColumns:
        data = detectAndRemoveOutliers(data, outlierColumns)

    if labelEncodingFeatures:
        data = labelEncoded(data, labelEncodingFeatures)

    if oneHotEncodingFeatures:
        data = oneHotEncoded(data, oneHotEncodingFeatures)

    if normalizeFeatures:
        data = normalizeData(data, normalizeFeatures)

    if bprintNullValues:
        data = printNullValues(data)
    
    if bprintData:
        printAndWriteFileWithDefinedFile(data)

    if bShowBoxplot:
        showBoxplot()


    printAndWriteFileWithDefinedFile("Finished Preprocessing")
    return data


######################### Other Stuff ######################### 

def showBoxplot(data, boxplot_features):
    data_long = pd.melt(data, value_vars=boxplot_features)
    sns.boxplot(x='variable', y='value', data=data_long)
    plt.xticks(rotation=45)
    plt.show()

def getColumns(data) -> list:
    return data.columns.tolist()

def removeColumns(columns: list, columnsToRemove: list) -> list:
    for r in columnsToRemove:
        columns.remove(r)
    return columns

def printAndWriteFileWithDefinedFile(content):
    printAndWriteInFile(content, "Logs//Preprocessing.txt")