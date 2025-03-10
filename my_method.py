from flask import Flask, render_template, request
import pickle
import difflib
import pandas as pd
import requests

app = Flask(__name__)

with open('movies.pkl', 'rb') as f:
    movies_data = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

TMDB_API_KEY = 'f10bc406937f3c8db7cd9e58d49b5347'  
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

def fetch_movie_poster(movie_title):
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    response = requests.get(search_url, params=params)
    data = response.json()
    
    if data['results']:
        poster_path = data['results'][0]['poster_path']
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return "https://via.placeholder.com/500x750?text=Poster+Not+Available"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not find_close_match:
            return render_template('index1.html', error="Movie not found. Please try another name.")
        
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movies[:10]):
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            poster_url = fetch_movie_poster(title_from_index)
            recommended_movies.append({
                'title': title_from_index,
                'poster_url': poster_url
            })
        
        return render_template('index1.html', movie_name=movie_name, recommended_movies=recommended_movies)
    
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
