import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore


def load_data(filepath):
    return pd.read_csv(filepath)

def remove_duplicates(data):
    data.drop_duplicates(inplace=True)
    return data

def check_null_values(data):
    print("Null Werte Absolut:")
    print(data.isnull().sum())
    print("\nNull Werte Prozentual:")
    print(data.isnull().mean() * 100)
    print("\n")
    
def detect_and_remove_outliers(data, columns):
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

def one_hot_encode_bp_category(data):
    data = pd.get_dummies(data, columns=['bp_category'])
    encoder = LabelEncoder()
    data['bp_category'] = encoder.fit_transform(data['bp_category'])
    return data

def normalize_data(data, selected_features):
    scaler = MinMaxScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])
    return data

def plot_boxplot(data):
    data_long = pd.melt(data, value_vars=['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'bmi'])
    sns.boxplot(x='variable', y='value', data=data_long)
    plt.xticks(rotation=45)
    plt.show()

def preprocessing(filepath):
    data = load_data(filepath)

    data = remove_duplicates(data)

    check_null_values(data)

    outlier_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

    data = detect_and_remove_outliers(data, outlier_columns)

    data = one_hot_encode_bp_category(data)

    selected_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

    data = normalize_data(data, selected_features)

    plot_boxplot(data)

    return data
