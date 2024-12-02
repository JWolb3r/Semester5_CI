
from sklearn.feature_selection import SequentialFeatureSelector # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("cardio_data_processed.csv")



