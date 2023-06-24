# web_scraping.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup

def scrape_job_data():
    # Define the URL of the website you want to scrape
    url = "https://in.indeed.com/"

    # Send an HTTP GET request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the relevant elements in the HTML structure and extract the necessary data
        job_listings = soup.find_all("div", class_="job-listing")
        
        job_data = []
        for job in job_listings:
            # Extract job details such as title, description, skills, etc.
            title = job.find("h2", class_="job-title").text.strip()
            description = job.find("div", class_="job-description").text.strip()
            skills = job.find("ul", class_="job-skills").text.strip()
            
            # Create a dictionary to store the extracted data
            job_info = {
                "title": title,
                "description": description,
                "skills": skills
            }
            
            # Append the job information to the list
            job_data.append(job_info)

        return job_data

    else:
        print("Failed to retrieve job data. Status code:", response.status_code)
        return None

def clean_data(raw_data):
    # Remove punctuation
    raw_data = [re.sub(f"[{string.punctuation}]", "", item) for item in raw_data]
    
    # Convert to lowercase
    raw_data = [item.lower() for item in raw_data]
    
    # Tokenize the text
    tokenized_data = [word_tokenize(item) for item in raw_data]
    
    # Remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    tokenized_data = [[word for word in item if word not in stop_words] for item in tokenized_data]
    
    # Lemmatization
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    tokenized_data = [[lemmatizer.lemmatize(word) for word in item] for item in tokenized_data]
    
    # Join the tokens back into sentences
    cleaned_data = [' '.join(item) for item in tokenized_data]
    
    return cleaned_data

def collect_and_preprocess_data():
    raw_data = scrape_job_data()
    cleaned_data = clean_data(raw_data)
    return cleaned_data
