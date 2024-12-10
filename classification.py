from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

def classify(hidden_layer, data, features, label):
    print("Start classification")
    

    x = data[features]
    y = data[label]

    x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.3, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=1000)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negativ","Positiv"])
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)
    return accuracy

if __name__ == "__main__":
    data = preprocessing("cardio_data_processed.csv")
    features = ['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','bmi','bp_category']
    label = 'cardio'
    classify([16,16,16], data, features, label)

