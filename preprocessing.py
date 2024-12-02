import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore


def preprocessing():
    data = pd.read_csv("cardio_data_processed.csv")

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    print("Null Werte Absolut:")
    print(data.isnull().sum())
    print("")
    print("Null Werte Prozentual:")
    print(data.isnull().mean()*100)
    print("")

    # Count values out of bounds
    total_outliers = 0

    for column in ['age_years','height','weight','ap_hi','ap_lo','bmi']:
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

        # Remove Outliers
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    print(f"Gesamtzahl der Ausreißer über alle Spalten: {total_outliers}")

    # One-Hot-Encoding
    data = pd.get_dummies(data, columns=['bp_category'])
    encoder = LabelEncoder()
    data['bp_category'] = encoder.fit_transform(data['bp_category'])

    # Normalizing
    selected_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    scaler = MinMaxScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])

    # Boxplot
    data_long = pd.melt(data, value_vars=['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'bmi'])
    sns.boxplot(x='variable', y='value', data=data_long)
    plt.xticks(rotation=45)  
    plt.show()

    return data