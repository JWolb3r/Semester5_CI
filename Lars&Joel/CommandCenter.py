from EvolutionaryAlgorithm import evolve
from FeatureSelection import ffs
from NN import trainAndTestMLP
from preprocessing import getColumns, preprocessing, removeColumns


filepath = "dataset\\cardio_data_processed.csv"


data = preprocessing(filepath, oneHotEncodingFeatures=["bp_category"], normalizeFeatures=["age", "height", "weight", "ap_hi", "ap_lo", "bmi"])

columns = removeColumns(getColumns(data), ["id", "age_years", "bp_category_encoded", "cardio"] )

bestFeatures = ffs(data, columns, "cardio", maxIter=1000)
bestFeatures.append("cardio")

print(trainAndTestMLP(data[bestFeatures], "cardio"))

evolve(data[bestFeatures], "cardio")