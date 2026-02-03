import pandas as pd
import numpy as np
import requests
import sklearn
import time


# 1) Load Data
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

# Rename so merge works
credits = credits.rename(columns={"movie_id": "id"})

# 2) Merge movies + credits
movies = movies.merge(credits, on="id")

# 3) Drop unneeded columns
print("Columns before dropping:", movies.columns)
movies = movies.drop(columns=[
    "homepage",
    "status",
    "title_x",
    "title_y",
    "production_countries"
], errors='ignore')

# Inspect data
print("Movies shape:", movies.shape)
print(movies.head())

# 4) Handle missing overviews
movies["overview"] = movies["overview"].fillna("")

# 5) Create TF-IDF Matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    min_df=3,         # ignore rare words
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(movies["overview"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 6) Compute Similarity (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 7) Build Reverse Index (title -> index)
indices = pd.Series(movies.index, index=movies["original_title"]).drop_duplicates()

# 8) Recommendation function
def give_recommendations(title, sim=cosine_sim):
    if title not in indices:
        return f"Movie '{title}' not found in database 😔"

    idx = indices[title]

    # get similarity scores
    sim_scores = list(enumerate(sim[idx]))

    # sort highest to lowest
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # take top 10 but skip itself
    sim_scores = sim_scores[1:11]

    # movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return titles
    return list(movies["original_title"].iloc[movie_indices])

# 9) Test it
import streamlit as st

st.title("Movie Recommendation System 🎬")
st.write("Enter a movie name to get recommendations:")

user_movie = st.text_input("Movie Name", "")

if user_movie:
    results = give_recommendations(user_movie)
    if isinstance(results, str):
        st.warning(results)
    else:
        st.subheader(f"Recommendations similar to '{user_movie}':")
        TMDB_API_KEY = "9b73be72146c5eceaf19cb20456a43a9"  # <-- Updated with your actual TMDB API key
        def fetch_poster_url(title, retries=3, delay=1):
            """Fetch poster URL from TMDB API given a movie title, with retry logic."""
            url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "query": title
            }
            for attempt in range(retries):
                try:
                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results")
                        if results:
                            poster_path = results[0].get("poster_path")
                            if poster_path:
                                return f"https://image.tmdb.org/t/p/w500{poster_path}"
                    else:
                        print(f"TMDB API error: {response.status_code}")
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error fetching poster for '{title}': {e}")
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching poster for '{title}': {e}")
                time.sleep(delay)  # Wait before retrying
            return None
        pass
        for r in results:
            # poster_path = movies.loc[movies["original_title"] == r, "poster_path"].values
            poster_url = fetch_poster_url(r)
            if poster_url:
                st.image(poster_url, width=150, caption=r)
            else:
                st.write("➤", r)
