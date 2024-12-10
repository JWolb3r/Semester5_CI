
from sklearn.feature_selection import SequentialFeatureSelector # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from preprocessing import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector 
from sklearn.model_selection import train_test_split


def ffs(data, features, label):
    print("Start feature selection forward")
    x = data[features]
    y = data[label]

    x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.3, random_state=42)

    model = MLPClassifier()

    selector = SequentialFeatureSelector(model,
                                        k_features=4,
                                        forward=True,
                                        floating=True,
                                        scoring='accuracy',
                                        cv=5)

    selector.fit(x_train, y_train)

    best_features = selector.k_feature_names_
    score = selector.k_score_

    print(f"Beste Features: {best_features}")
    print(f"CV-Score: {score}")

if __name__ == "__main__":
    data = preprocessing("cardio_data_processed.csv")
    features = ['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','bmi','bp_category']
    label = 'cardio'

