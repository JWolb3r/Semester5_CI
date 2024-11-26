import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")

x = data[["policy_tenure", "age_of_car", "fuel_type", "displacement", "cylinder", "is_brake_assist"]]
y = data["is_claim"]  

label_encoder = LabelEncoder()

x["fuel_type"] = label_encoder.fit_transform(x['fuel_type']).astype(float)
x["is_brake_assist"] = label_encoder.fit_transform(x['is_brake_assist']).astype(float)

scaler = StandardScaler()

numerical_columns = ['policy_tenure', 'age_of_car', 'displacement', 'cylinder']

x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

sumAccuracy = 0

for zahl in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    nn = MLPClassifier(hidden_layer_sizes=(20, 20))
    nn.fit(x_train, y_train)
    y_pred = nn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    sumAccuracy += accuracy
    print(f"Accuracy: {accuracy:.2f}")


print(f"SumAccuracy: {sumAccuracy/100:.2f}")
