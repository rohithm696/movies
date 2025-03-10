from flask import Flask, render_template, request, redirect, url_for
import difflib
import pandas as pd
import requests
from urllib.parse import quote_plus, unquote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the movies dataset
movies_data = pd.read_csv('movies.csv')

# Ensure 'title' column exists in dataset
if 'title' not in movies_data.columns:
    raise ValueError("Error: 'title' column missing from movies.csv")

# Select features for recommendations
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace NaN values with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine all features into a single string column
movies_data['combined_features'] = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

# Initialize the TF-IDF Vectorizer and compute feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute the cosine similarity matrix
similarity = cosine_similarity(feature_vectors)

# TMDB API key and base URL (Replace with your actual API key)
TMDB_API_KEY = 'f10bc406937f3c8db7cd9e58d49b5347'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

def fetch_movie_details(movie_title):
    """Fetch movie details from TMDB API."""
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {'api_key': TMDB_API_KEY, 'query': movie_title}
    response = requests.get(search_url, params=params)
    data = response.json()

    if data.get('results'):
        movie_id = data['results'][0]['id']
        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        details_params = {'api_key': TMDB_API_KEY, 'append_to_response': 'credits'}
        details_response = requests.get(details_url, params=details_params)
        details_data = details_response.json()

        return {
            'poster_url': f"https://image.tmdb.org/t/p/w500{details_data.get('poster_path', '')}" if details_data.get('poster_path') else "https://via.placeholder.com/500x750?text=Poster+Not+Available",
            'title': details_data.get('title', 'N/A'),
            'runtime': details_data.get('runtime', 'N/A'),
            'genres': ', '.join([genre['name'] for genre in details_data.get('genres', [])]),
            'cast': ', '.join([actor['name'] for actor in details_data.get('credits', {}).get('cast', [])[:5]])
        }

    return {
        'poster_url': "https://via.placeholder.com/500x750?text=Poster+Not+Available",
        'title': 'N/A',
        'runtime': 'N/A',
        'genres': 'N/A',
        'cast': 'N/A'
    }

def get_recommendations(movie_name):
    """Finds and returns recommended movies based on similarity scores."""
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None, None  

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    user_movie_details = fetch_movie_details(close_match)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:11]):  # Get Top 10 recommendations
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommended_movies.append(fetch_movie_details(title_from_index))

    return user_movie_details, recommended_movies

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main search form and redirects to the recommendation page."""
    if request.method == "POST":
        movie_name = request.form.get("movie_name", "").strip()
        if not movie_name:
            return render_template("index.html", error="Please enter a movie name.")
        return redirect(url_for("recommend", movie_name=quote_plus(movie_name)))  # Encode movie name safely
    return render_template("index.html")

@app.route("/recommend/<movie_name>", methods=["GET"])
def recommend(movie_name):
    """Fetches and displays recommendations for a given movie."""
    movie_name = unquote_plus(movie_name)  # Decode movie name
    user_movie_details, recommended_movies = get_recommendations(movie_name)

    if user_movie_details is None:
        return render_template("index.html", error="Movie not found! Please try another name.")

    return render_template("index.html", user_movie_details=user_movie_details, recommended_movies=recommended_movies)

if __name__ == "__main__":
    app.run(debug=True)
