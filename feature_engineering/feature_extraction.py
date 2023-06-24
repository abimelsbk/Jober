# feature_extraction.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def extract_features(data):
 def extract_features(data):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
    
    # Fit the vectorizer on the preprocessed data
    features = vectorizer.fit_transform(data)
    
    # Convert the features to a dense matrix representation
    features = features.toarray()
    
    return features

def engineer_features(data):
   def engineer_features(features):
    # Example feature engineering
    # Calculate additional statistical features
    
    # Mean of each feature
    mean_features = np.mean(features, axis=1)
    
    # Standard deviation of each feature
    std_features = np.std(features, axis=1)
    
    # Maximum value of each feature
    max_features = np.max(features, axis=1)
    
    # Concatenate the additional features with the original features
    engineered_features = np.column_stack((features, mean_features, std_features, max_features))
    
    return engineered_features

def perform_feature_engineering(data):
    extracted_features = extract_features(data)
    engineered_features = engineer_features(extracted_features)
    return engineered_features
