# model_training.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    #Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Random Forest classifier
    model = RandomForestClassifier()
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation data
    accuracy = model.score(X_val, y_val)
    print("Validation Accuracy:", accuracy)
    
    return model
    # model = RandomForestClassifier()
    # model.fit(X, y)
    # return model

def train_and_evaluate_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = train_model(X_train, y_train)
    return model
