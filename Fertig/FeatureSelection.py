from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier


def ffs(datasetName, data, features, label, hiddenLayerSizes=(8,8,8), maxIter=4000, randomState=42, activation="relu", returnFeatures = 4, crossValidation = 5):
    print(f"Start forward feature selection of file {datasetName}")

    x = data[features]
    y = data[label]
 
    model = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation=activation, max_iter=maxIter, random_state=randomState)
    selector = SequentialFeatureSelector(model,
                                        n_features_to_select=returnFeatures,
                                        scoring='accuracy',
                                        cv=crossValidation)
    selector.fit(x, y)

    return [features[i] for i in selector.get_support(indices=True)]