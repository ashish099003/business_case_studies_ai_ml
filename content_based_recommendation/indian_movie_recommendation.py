# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)


def visualise_ratings(ratings_df):
    # Visualize rating distribution
    plt.figure(figsize=(12, 5))

    # Plot 1: Rating distribution
    plt.subplot(1, 2, 1)
    ratings_df['rating'].hist(bins=10, edgecolor='black', alpha=0.7, color='#FF6B6B')
    plt.title('Distribution of Movie Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Plot 2: Number of ratings per movie
    plt.subplot(1, 2, 2)
    ratings_per_movie = ratings_df.groupby('movieId').size()
    plt.hist(ratings_per_movie, bins=50, edgecolor='black', alpha=0.7, color='#4ECDC4')
    plt.title('Distribution of Ratings per Movie', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def clean_list_string(s):
    if isinstance(s, str):
        # Remove brackets and quotes, convert to space-separated string
        s = s.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
        return s.strip()
    return str(s)

def create_tfidf_matrix(movies_df):
    # Create TF-IDF matrix for content similarity
    print("🔤 Creating TF-IDF matrix for content-based filtering...")

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',

    )

    # Fit and transform the combined features
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

    print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of movies: {tfidf_matrix.shape[0]}")
    print(f"Number of features: {tfidf_matrix.shape[1]}")

    return tfidf_matrix

def calculate_similarity_between_movies(tfidf_matrix):
    # Calculate cosine similarity between movies
    print("📐 Calculating cosine similarity between movies...")

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"✅ Similarity matrix shape: {cosine_sim.shape}")
    print(f"\nSample similarities for first movie:")
    print(f"Min similarity: {cosine_sim[0].min():.3f}")
    print(f"Max similarity: {cosine_sim[0].max():.3f}")
    print(f"Mean similarity: {cosine_sim[0].mean():.3f}")
    return cosine_sim

def get_content_based_recommendations(movie_title, movies_df, cosine_sim, n_recommendations=10):
    """
    Get content-based movie recommendations

    Parameters:
    -----------
    movie_title : str
        Title of the movie to base recommendations on
    movies_df : DataFrame
        Movies dataframe
    cosine_sim : array
        Cosine similarity matrix
    n_recommendations : int
        Number of recommendations to return

    Returns:
    --------
    DataFrame with recommended movies
    """
    try:
        # Get movie index
        idx = movies_df[movies_df['name'].str.contains(movie_title, case=False)].index[0]

        # Get similarity scores for all movies with this movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get indices of most similar movies (excluding the movie itself)
        movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]

        # Return the top similar movies
        recommendations = movies_df.iloc[movie_indices][['movieId', 'name', 'genre']].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores[1:n_recommendations+1]]

        return recommendations

    except IndexError:
        print(f"❌ Movie '{movie_title}' not found in database")
        return pd.DataFrame()

def load_n_preprocess_movie():
    movies_df = pd.read_csv("movies.csv")
    print("✅ Movies data loaded successfully!")

    # Load ratings data with lines=True for JSONL format
    ratings_df_raw = pd.read_json('ratings.json', lines=True)
    print("✅ Raw ratings data loaded successfully!")
    print(ratings_df_raw.head())
    ratings_list = []
    for _, row in ratings_df_raw.iterrows():
        user_id = row['_id']
        if 'rated' in row and isinstance(row['rated'], dict):
            for movie_id, rating_value in row['rated'].items():
                if movie_id != 'submit' and isinstance(rating_value, list) and len(rating_value) > 0:
                    try:
                        rating = float(rating_value[0])
                        # Convert -1,0,1 ratings to 1-5 scale
                        if rating == -1:
                            rating = 1
                        elif rating == 0:
                            rating = 3
                        elif rating == 1:
                            rating = 5
                        ratings_list.append({
                            'userId': user_id,
                            'movieId': movie_id,
                            'rating': rating
                        })
                    except:
                        pass

    ratings_df = pd.DataFrame(ratings_list)
    print("✅ Ratings data transformed successfully!")

    # Load users data
    users_df = pd.read_csv('users.csv')
    print("✅ Users data loaded successfully!")

    # visualise_ratings(ratings_df)

    movies_df = movies_df.fillna('')
    movies_df['movieId'] = movies_df['movie_id']

    # Create a feature string combining available text features
    movies_df['combined_features'] = ''

    if 'name' in movies_df.columns:
        movies_df['combined_features'] += movies_df['name'].astype(str) + ''

    # Add genre
    if 'genre' in movies_df.columns:
        movies_df['combined_features'] += movies_df['genre'].apply(clean_list_string) + ' '

    # Add cast
    if 'cast' in movies_df.columns:
        movies_df['combined_features'] += movies_df['cast'].apply(clean_list_string) + ' '

    # Add director
    if 'director' in movies_df.columns:
        movies_df['combined_features'] += movies_df['director'].apply(clean_list_string) + ' '

    # Add description
    if 'description' in movies_df.columns:
        movies_df['combined_features'] += movies_df['description'].astype(str)

    # Clean up the combined features
    movies_df['combined_features'] = movies_df['combined_features'].str.strip()

    print("Sample combined features:")
    for i in range(min(3, len(movies_df))):
        print(f"\nMovie {i + 1}: {movies_df['name'].iloc[i]}")
        print(f"Features: {movies_df['combined_features'].iloc[i][:200]}...")

    tfidf_matrix = create_tfidf_matrix(movies_df)
    cos_sim = calculate_similarity_between_movies(tfidf_matrix)
    print("🎬 CONTENT-BASED RECOMMENDATIONS DEMO")
    print("=" * 50)

    # Pick a popular movie (adjust based on your dataset)
    test_movie = movies_df['name'].iloc[0]  # Use first movie as example
    print(f"\n🎯 Finding movies similar to: '{test_movie}'")

    recommendations = get_content_based_recommendations(test_movie, movies_df, cos_sim)
    print("\n📋 Top 10 Recommendations:")
    print(recommendations)




if __name__=='__main__':
    load_n_preprocess_movie()
