import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoostClassifier" : KNeighborsClassifier(),
    "GaussianProcessClassifier" : GaussianProcessClassifier(),
    "MLPClassifier" : MLPClassifier(),
    "GaussianNB" : GaussianNB()
}

def ClassifierSelection(models, X_train, y_train, y_test):
    for modelName, model in models.items():
        print(f"Training model: {modelName}")
        model.fit(X_train, y_train)
    predictions = {modelName:model.predict(X_test) for modelName, model in models.items()}
    df = pd.DataFrame(predictions)
    df["gt"] = y_test.reset_index(drop=True)
    print(f"Accuracy: {round(accuracy_score(y_test,prediction),3)}")
    print(f"Precision: {round(precision_score(y_test,prediction,average='weighted'),3)}")
    print(f"Recall:, {round(recall_score(y_test,prediction,average='weighted'),3)}\n")

def verifyPredictions(y_test, prediction):
    print("Accuracy", accuracy_score(y_test, prediction))
    print("Precision", precision_score(y_test, prediction,average='weighted'))
    print("Recall", recall_score(y_test, prediction,average='weighted'))