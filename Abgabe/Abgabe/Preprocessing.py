import pandas as pd 
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utilty import printAndWriteInFile, printAndWriteInPreprocessingFile
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore


def loadData(filepath):
    printAndWriteInPreprocessingFile("\nLoad Data")
    with open(filepath, 'r', encoding='utf-8') as file:
        first_line = file.readline()
        
    if ';' in first_line:
        separator = ';'
    else:
        separator = ','
    return pd.read_csv(filepath, sep=separator)

def removeDuplicates(data):
    printAndWriteInPreprocessingFile("\nRemove Duplicates")
    data.drop_duplicates(inplace=True)
    return data

def printNaNValues(data):
    printAndWriteInPreprocessingFile("\nPrint NaN Values")
    printAndWriteInPreprocessingFile("NaN values (absolute):")
    printAndWriteInPreprocessingFile(data.isna().sum().to_string()) 
    printAndWriteInPreprocessingFile("\n")

    printAndWriteInPreprocessingFile("NaN values (percentage):")
    printAndWriteInPreprocessingFile(data.isna().mean().mul(100).round(2).to_string()) 
    printAndWriteInPreprocessingFile("\n")

    return data

def deleteNaNValues(data):
    printAndWriteInPreprocessingFile("\nDelete NaN Values")
    printAndWriteInPreprocessingFile(f"Length of dataset (before NaN removal): {len(data)}\n")
    data = data.dropna(axis=0, how="any")
    printAndWriteInPreprocessingFile(f"Length of dataset (after NaN removal): {len(data)}\n")
    return data

    
def detectAndRemoveOutliers(data, columns):
    printAndWriteInPreprocessingFile("\nDetect and remove Outliers")
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
        
        printAndWriteInPreprocessingFile(f"Spalte: {column}")
        printAndWriteInPreprocessingFile(f"Unterer Whisker: {lower_bound}")
        printAndWriteInPreprocessingFile(f"Oberer Whisker: {upper_bound}")
        printAndWriteInPreprocessingFile(f"Anzahl der Ausreißer in {column}: {outliers_count}")
        printAndWriteInPreprocessingFile(f"Größter Ausreißer in {column}: {largest_outlier}")
        printAndWriteInPreprocessingFile(f"Kleinster Ausreißer in {column}: {smallest_outlier}")
        printAndWriteInPreprocessingFile(f"Mittelwert von {column}: {mean_value}")
        printAndWriteInPreprocessingFile("-" * 50)

        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    printAndWriteInPreprocessingFile(f"Gesamtzahl der Ausreißer über alle Spalten: {total_outliers}")
    return data

def labelEncoded(data,values):
    printAndWriteInPreprocessingFile("\nLabel Encode")
    for a in values:
        encoder = LabelEncoder()
        data[a] = encoder.fit_transform(data[a])
    return data

def oneHotEncoded(data, columns: list):
    printAndWriteInPreprocessingFile("\nOneHotEncode")
    return pd.get_dummies(data, columns=columns)

def normalizeData(data, selected_features: list):
    printAndWriteInPreprocessingFile("\nNormalize Data")
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
        bPrintNanValues = True,
        bDeleteNanValues = False,
        bShowBoxplot = False,
        bPrintInfo = True
        ):

    printAndWriteInPreprocessingFile(f"\nStart preprocessing of file {filepath}")


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

    if bPrintNanValues:
        data = printNaNValues(data)

    if bDeleteNanValues:
        data = deleteNaNValues(data)

    if bPrintInfo:
        printLengthAndColumns(data)

    if bShowBoxplot:
        showBoxplot()



    printAndWriteInPreprocessingFile("Finished Preprocessing")
    return data


######################### Other Stuff ######################### 

def showBoxplot(data, boxplot_features):
    printAndWriteInPreprocessingFile("\nShow Boxplot")
    data_long = pd.melt(data, value_vars=boxplot_features)
    sns.boxplot(x='variable', y='value', data=data_long)
    plt.xticks(rotation=45)
    plt.show()

def getColumns(data) -> list:
    return data.columns.tolist()

def removeColumns(columns: list, columnsToRemove: list) -> list:
    printAndWriteInPreprocessingFile("\nRemove Columns")
    for r in columnsToRemove:
        columns.remove(r)
    return columns

def printLengthAndColumns(data: pd.DataFrame, ):
    printAndWriteInPreprocessingFile("\nPrint Datset Infos")
    printAndWriteInPreprocessingFile(f"\nLength of Dataset: {len(data)}")
    printAndWriteInPreprocessingFile(f"\nColumns of Dataset: {data.columns}")