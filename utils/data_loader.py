import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_movies(data_path):
    """Load movies data from the ML-1M dataset with proper encoding."""
    columns = ["movie_id", "title", "genres"]
    
    # Load the movies data with 'latin-1' encoding (also known as 'latin1')
    movies = pd.read_csv(f"{data_path}/movielens/movies.dat", sep="::", names=columns, engine="python", encoding='latin-1')
    
    # Split genres by '|'
    # movies["genres"] = movies["genres"].str.split("|")
    
    return movies

def load_users(data_path):
    """Load users data from the ML-1M dataset with proper encoding."""
    columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    
    # Load the users data with 'latin-1' encoding
    users = pd.read_csv(f"{data_path}/movielens/users.dat", sep="::", names=columns, engine="python", encoding='latin-1')
    
    return users

def load_ratings(data_path):
    """Load ratings data from the ML-1M dataset with proper encoding."""
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    
    # Load the ratings data with 'latin-1' encoding
    ratings = pd.read_csv(f"{data_path}/movielens/ratings.dat", sep="::", names=columns, engine="python", encoding='latin-1')
    
    return ratings

def merge_data(movies, users, ratings):
    """Merge movies, users, and ratings data into one DataFrame."""
    # Merge ratings with movie titles and genres
    ratings_with_titles = pd.merge(ratings, movies[['movie_id', 'title', 'genres']], on='movie_id', how='left')
    
    # Merge with user information
    complete_data = pd.merge(ratings_with_titles, users[['user_id', 'gender', 'age', 'occupation']], on='user_id', how='left')
    
    return complete_data

def process_genres(movies):
    # Split the genres in the 'genres' column (separated by '|')
    genre_lists = movies['genres'].str.split('|')

    # Use MultiLabelBinarizer to one-hot encode the genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genre_lists)

    # Convert the matrix back to a DataFrame with column names as genre labels
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
    return genre_df

def preprocess_data(data):
    """Preprocess the ML-1M data (handle missing values, normalize ratings)."""
    # Drop rows with missing values
    data = data.dropna()
    
    # Normalize ratings between 0 and 1 (min-max normalization)
    data['rating'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())
    
    return data

def save_processed_data(processed_data, file_path):
    """Save the preprocessed data to a CSV file."""
    processed_data.to_csv(file_path, index=False)

def get_user_preferences(user_id, merged_data):
    # Initialize the user_preferences dictionary
    user_preferences = {
        "liked_genres": [],
        "disliked_genres": [],
        "watch_history": []
    }
    
    # Filter data for the specific user
    user_data = merged_data[merged_data['user_id'] == user_id][['title', 'rating', 'liked', 'genres_list']]
    
    # Dictionary to count likes and dislikes for each genre
    genre_counts = {}
    
    # Iterate through the user's movie data
    for index, row in user_data.iterrows():
        for genre in row['genres_list']:
            if genre not in genre_counts:
                genre_counts[genre] = {'likes': 0, 'dislikes': 0}
            if row['liked']:
                genre_counts[genre]['likes'] += 1
            else:
                genre_counts[genre]['dislikes'] += 1
        
        # Add to watch history (including genres)
        user_preferences['watch_history'].append({
            "title": row['title'],
            "liked": row['liked'],
            "genres": row['genres_list']  # Include genres here
        })
    
    # Identify conflicting genres
    conflicting_genres = {genre for genre, counts in genre_counts.items() if counts['likes'] > 0 and counts['dislikes'] > 0}
    
    # Separate liked and disliked genres
    liked_genres = {genre: counts['likes'] for genre, counts in genre_counts.items() if counts['likes'] > 0 and genre not in conflicting_genres}
    disliked_genres = {genre: counts['dislikes'] for genre, counts in genre_counts.items() if counts['dislikes'] > 0 and genre not in conflicting_genres}
    
    # Add conflicting genres to disliked_genres or liked_genres based on frequency
    for genre in conflicting_genres:
        if genre_counts[genre]['likes'] > genre_counts[genre]['dislikes']:
            liked_genres[genre] = genre_counts[genre]['likes']
        else:
            disliked_genres[genre] = genre_counts[genre]['dislikes']
    
    # Sort genres by frequency
    sorted_liked_genres = sorted(liked_genres.items(), key=lambda x: x[1], reverse=True)
    sorted_disliked_genres = sorted(disliked_genres.items(), key=lambda x: x[1], reverse=True)
    
    # Extract all liked and disliked genres (no top_k limit)
    user_preferences['liked_genres'] = [genre for genre, _ in sorted_liked_genres]
    user_preferences['disliked_genres'] = [genre for genre, _ in sorted_disliked_genres]
    
    return user_preferences

def get_item_features(movie_id, movies_data):
    """
    Returns the features of a movie given its movie_id.

    Parameters:
    - movie_id (int): The ID of the movie.
    - movies_data (DataFrame): The DataFrame containing movie information.

    Returns:
    - dict: A dictionary with the movie's title, genres (as a string), and keywords (if available).
    """
    # Filter the movie data based on movie_id
    movie = movies_data[movies_data['movie_id'] == movie_id]
    
    # Check if the movie exists
    if movie.empty:
        return f"Movie with ID {movie_id} not found."
    
    # Extract features
    title = movie.iloc[0]['title']
    
    # Convert genres_list to a comma-separated string
    genres_list = movie.iloc[0]['genres_list']
    genres = ", ".join(genres_list) if isinstance(genres_list, list) else genres_list
    
    # Check for keywords if available
    keywords = movie.iloc[0]['keywords'] if 'keywords' in movie.columns else []

    # Create the feature dictionary
    features = {
        "title": title,
        "genre": genres,
        # "keywords": keywords if isinstance(keywords, list) else []
    }
    
    return features