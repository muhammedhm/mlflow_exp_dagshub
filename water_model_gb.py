import pandas as pd
import numpy as np
from dvclive import Live
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='muhammedhm', repo_name='mlflow_exp_dagshub', mlflow=True)

mlflow.set_experiment("Water_Potability_Experiment_2")
mlflow.set_tracking_uri("https://dagshub.com/muhammedhm/mlflow_exp_dagshub.mlflow")

data = pd.read_csv("./data/water_potability.csv")


train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 500

with mlflow.start_run() as run:

    mlflow.log_param("n_estimators", n_estimators)

    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    # save 
    pickle.dump(clf, open("model.pkl", "wb"))


    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values



    model = pickle.load(open('model.pkl',"rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_score = f1_score(y_test,y_pred)

# with Live(save_dvc_exp=True) as live:
#     live.log_metric("acc",acc)
#     live.log_metric("precision", precision)
#     live.log_metric("recall", recall)
#     live.log_metric("f1-score",f1_score)

#     live.log_param("n_estimators",n_estimators)
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1_score)

    mlflow.log_param("n_estimators", n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    mlflow.sklearn.log_model(clf, "GradientBoostingClassifier_model")

    mlflow.log_artifact(r"D:\Machine_learning\MLOps\exp_track\src\water_model_gb.py")

    mlflow.set_tag("model_type", "GradientBoostingClassifier")
    mlflow.set_tag("author", "Mhd")

    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 Score:",f1_score)