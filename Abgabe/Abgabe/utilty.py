import pandas as pd
import shutil
import os

class MetricsContainer:
    def __init__(self, acc, f1score):
        self.acc = acc
        self.f1score = f1score

def printAndWriteInFile(content, filename):
    print("\n")
    print(content)

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(filename, 'a') as file:
            file.write(str(content) + '\n')
    except Exception as e:
        print(f"Error updating the file: {e}")


def printAndWriteInPreprocessingFile(content):
    printAndWriteInFile(content, "Logs/Preprocessing.txt")

def printAndWriteInEAOptimizeFile(content):
    printAndWriteInFile(content, "Logs/NN_EA_OPTIMIZE.txt")

def printAndWriteInEAOptimizeFile(content):
    printAndWriteInFile(content, "Logs/NN_EA_OPTIMIZE.txt")

def printAndWriteInFileBestFeatures(content):
    printAndWriteInFile(content, "Logs/BestFeatures.txt")

def printAndWriteInFileAcc(content):
    printAndWriteInFile(content, "Logs/AccssOfAlgorithms.txt")

def printAndWriteInFileBestComb(content):
    printAndWriteInFile(content, "Logs/BestComb.txt")

def printAndWriteInFileAvgAcc(content):
    printAndWriteInFile(content, "Logs/AvgAcssOfAlgs.txt")

def printAndWriteInFileF1Score(content):
    printAndWriteInFile(content, "Logs/F1ScoreOfAlgs.txt")

def printAndWriteInAvgFileF1Score(content):
    printAndWriteInFile(content, "Logs/AvgF1ScoreOfAlgs.txt")

def combineFiles(trainPath, testPath, destinationPath):
    train_df = pd.read_csv(trainPath)
    test_df = pd.read_csv(testPath)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    combined_df.to_csv(destinationPath, index=False)

    print(combined_df.head())

