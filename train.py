import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

def evaluation_metrics(y_test , y_pred):
    rmse = mean_squared_error(y_test , y_pred , squared = False)
    mae = mean_absolute_error(y_test , y_pred)
    r2 = r2_score(y_test , y_pred)
    return rmse , mae , r2

if __name__ == "__main__":
    csv_link = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    dataset = pd.read_csv(csv_link , sep =";")
    X = dataset.drop(columns="quality")
    Y = dataset["quality"]

    X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size = 0.3 , random_state = 0 )

    with mlflow.start_run():
        alpha = 0.5
        l1_ratio = 0.5

        lr = ElasticNet(alpha = 0.1 , l1_ratio = 0.3 , random_state = 0)
        lr.fit(X_train , y_train)
        prediction = lr.predict(X_test)
        rmse , mae , r2 = evaluation_metrics(y_test , prediction)
        print(f"rmse is {rmse}")
        print(f"msa is {mae}")
        print(f"r2 score is {r2}")

        mlflow.log_param("alpha" , alpha)
        mlflow.log_param("l1_ratio" , l1_ratio)

        mlflow.log_metric("rmse" , rmse)
        mlflow.log_metric("mae" , mae)
        mlflow.log_metric("r2" , r2)

        mlflow.sklearn.log_model(lr ,"model")
