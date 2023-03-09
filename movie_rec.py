import pandas as pd

movies = pd.read_csv("ml-25m/movies.csv")
print(movies)

import re 
def clean_title(title):  #need to clean results to get rid of any extra characters such as parenthesis etc 

    return re.sub("[^a-zA-Z0-9 ]", "", title) # removes everything that isnt a lowercase/upper case or number/space


movies["clean_title"] = movies["title"].apply(clean_title)
print(movies)


from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # instead of individually searching for a word, searches for pairs.

tfidf = vectorizer.fit_transform(movies["clean_title"])

#to compute the similarity between two titles  

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = "Toy story 1995"
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    #find similarity with all titles in data and search term
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    #titles that have the greatest similarity to our search term 
    indices  = np.argpartition(similarity, -5)[-5:]

    results = movies.iloc[indices][::-1]
    return results

import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value= "Toy Story",
    description = "Movie Title:",
    disabled = False
)
movie_input