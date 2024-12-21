import pandas as pd

import os

def printAndWriteInFile(content, filename):
    print(content)

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    try:
        with open(filename, 'a') as file:
            if isinstance(content, list):
                for element in content:
                    file.write(str(element) + '\n')
                file.write('\n')
            elif isinstance(content, object):
                for key, value in obj.__dict__.items():
                    file.write("Key: " + key + " Value: " + value)
                file.write('\n')
            else:
                file.write(str(content) + '\n')
                file.write('\n')
        print(f"The file '{filename}' was successfully updated.")
    except Exception as e:
        print(f"Error updating the file: {e}")


def combineFiles(trainPath, testPath, destinationPath):
    train_df = pd.read_csv(trainPath)
    test_df = pd.read_csv(testPath)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    combined_df.to_csv(destinationPath, index=False)

    print(combined_df.head())