import pandas as pd

train_path = "DataSets/titanic_train.csv"
test_path = "DataSets/titanic_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

combined_df = pd.concat([train_df, test_df], ignore_index=True)

combined_df.to_csv("DataSets/titanic_combined.csv", index=False)

print(combined_df.head())


