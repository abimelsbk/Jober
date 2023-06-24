import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_data(job_data):
    # Code to preprocess the collected job data
    preprocessed_data = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    for job in job_data:
        # Preprocess job description
        description = job['description']
        description = re.sub(r'[^\w\s]', '', description)  # Remove punctuation
        description = description.lower()  # Convert to lowercase
        tokens = word_tokenize(description)  # Tokenize text
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
        preprocessed_description = ' '.join(tokens)
        
        # Add preprocessed data to the list
        preprocessed_job = {
            'title': job['title'],
            'company': job['company'],
            'description': preprocessed_description
        }
        preprocessed_data.append(preprocessed_job)
    
    return preprocessed_data
