# model_evaluation.py

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    # Code to evaluate the model's performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
