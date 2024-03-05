import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore") 

from sklearn.linear_model import (
    RANSACRegressor,
    HuberRegressor,
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('data/felicidad_corrupt.csv')
    print(df.head())

    # Split the dataset into features (X) and target (y)
    X = df.drop(['country', 'score'], axis=1)
    y = df[['score']]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the regression models to use
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimators.items():
     estimador.fit(X_train, y_train)
    predictions = estimador.predict(X_test)
    print("=" * 32)
    print(name)
    plt.ylabel('Predicted Score')
    plt.xlabel('Real Score')
    plt.title('Predicted VS Real')
    plt.scatter(y_test, predictions)
    plt.plot(predictions, predictions,'r--')
    plt.show()