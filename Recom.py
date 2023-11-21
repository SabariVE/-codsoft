import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import numpy as np

# Load the MovieLens dataset (Assuming it's already downloaded)
movies = pd.read_csv('movies.Leo')
ratings = pd.read_csv('ratings.4.1')

# Merge the ratings and movies dataframes
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_movie_ratings)

# Convert the similarity matrix into a dataframe
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

# Function to get movie recommendations for a given user
def get_movie_recommendations(user_id):
    # Get the movies the user has already rated
    user_rated_movies = user_movie_ratings.loc[user_id]

    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]

    # Weighted sum of ratings by similar users
    weighted_sum = user_movie_ratings.loc[similar_users.index] * similar_users.values[:, np.newaxis]

    # Sum the weighted ratings and normalize
    recommendation_scores = weighted_sum.sum(axis=0) / (similar_users.abs().sum() + 1e-6)

    # Filter out movies the user has already rated
    recommendations = recommendation_scores.drop(user_rated_movies.index)

    # Sort the recommendations by score
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations

# Example: Get movie recommendations for user 1
user_id = 1
recommendations = get_movie_recommendations(user_id)
print(f"Top 5 movie recommendations for user {user_id}:\n{recommendations.head(5)}")
