# %%
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
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names= 'value')

display(movie_input, movie_list)

    



# %%

#finding users who liked the same movie

ratings = pd.read_csv("ml-25m/ratings.csv")

ratings.dtypes

movie_id = 1
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 5 )]["userId"].unique()


#finds users that liked the same movie as search input
similar_users_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_users_recs

# %%
similar_users_recs= similar_users_recs.value_counts() / len(similar_users)

similar_users_recs= similar_users_recs[similar_users_recs > .1]

similar_users_recs

# %%
#how much do all users in our dataset like these movies.

all_users = ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["rating"] >4)]
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())



