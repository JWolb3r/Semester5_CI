import pandas as pd 
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utilty import printAndWriteInFile
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore


def loadData(filepath):
    print("\nLoad Data")
    return pd.read_csv(filepath)

def removeDuplicates(data):
    print("\nRemove Duplicates")
    data.drop_duplicates(inplace=True)
    return data

def printNaNValues(data):
    print("\nPrint NaN Values")
    print("NaN values (absolute):")
    print(data.isna().sum().to_string()) 
    print("\n")

    print("NaN values (percentage):")
    print(data.isna().mean().mul(100).round(2).to_string()) 
    print("\n")

    return data

def deleteNaNValues(data):
    print("\nDelete NaN Values")
    print(f"Length of dataset (before NaN removal): {len(data)}\n")
    data = data.dropna(axis=0, how="any")
    print(f"Length of dataset (after NaN removal): {len(data)}\n")
    return data

    
def detectAndRemoveOutliers(data, columns):
    print("\nDetect and remove Outliers")
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
        
        print(f"Spalte: {column}")
        print(f"Unterer Whisker: {lower_bound}")
        print(f"Oberer Whisker: {upper_bound}")
        print(f"Anzahl der Ausreißer in {column}: {outliers_count}")
        print(f"Größter Ausreißer in {column}: {largest_outlier}")
        print(f"Kleinster Ausreißer in {column}: {smallest_outlier}")
        print(f"Mittelwert von {column}: {mean_value}")
        print("-" * 50)

        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    print(f"Gesamtzahl der Ausreißer über alle Spalten: {total_outliers}")
    return data

def labelEncoded(data,values):
    print("\nLabel Encode")
    encoder = LabelEncoder()
    data[values] = encoder.fit_transform(data[values])
    return data

def oneHotEncoded(data, columns: list):
    print("\nOneHotEncode")
    return pd.get_dummies(data, columns=columns)

def normalizeData(data, selected_features: list):
    print("\nNormalize Data")
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

    print(f"\nStart preprocessing of file {filepath}")


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



    print("Finished Preprocessing")
    return data


######################### Other Stuff ######################### 

def showBoxplot(data, boxplot_features):
    print("\nShow Boxplot")
    data_long = pd.melt(data, value_vars=boxplot_features)
    sns.boxplot(x='variable', y='value', data=data_long)
    plt.xticks(rotation=45)
    plt.show()

def getColumns(data) -> list:
    return data.columns.tolist()

def removeColumns(columns: list, columnsToRemove: list) -> list:
    print("\nRemove Columns")
    for r in columnsToRemove:
        columns.remove(r)
    return columns

def printLengthAndColumns(data: pd.DataFrame):
    print("\nPrint Datset Infos")
    print(f"\nLength of Dataset: {len(data)}")
    print(f"\nColumns of Dataset: {data.columns}")