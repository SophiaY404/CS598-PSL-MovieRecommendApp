import pandas as pd
import numpy as np
import requests

def read_data():    
    # Read and clean up users data
    user_url = "https://liangfgithub.github.io/MovieData/users.dat?raw=true"
    user_response = requests.get(user_url)
    user_lines = user_response.text.split('\n')
    user_data = [line.split("::") for line in user_lines if line]
    users = pd.DataFrame(user_data, columns=['user_id', 'gender', 'age', 'occupation', 'zip'])
    users['user_id'] = users['user_id'].astype(int)    
    
    # Read and clean up movie data
    movie_url = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
    movie_response = requests.get(movie_url)
    movie_lines = movie_response.text.split('\n')
    movie_data = [line.split("::") for line in movie_lines if line]
    movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
    movies['movie_id'] = movies['movie_id'].astype(int)
    
    # Read and clean up ratings data
    rating_url = "https://liangfgithub.github.io/MovieData/ratings.dat?raw=true"
    rating_response = requests.get(rating_url)
    rating_lines = rating_response.text.split('\n')
    rating_data = [line.split("::") for line in rating_lines if line]
    ratings = pd.DataFrame(rating_data, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(int)
    return users, movies, ratings


# take the average of the ratings of the movie, and then multiply by log(# ratings for the movie).
def weighted_average(ratings):
    weighted_avg = (ratings['rating']).mean() * np.log(ratings['rating'].count()).sum()
    return weighted_avg

def regular_average(ratings): 
    return (ratings['rating'].sum())/ ratings['rating'].count()

def setup_recommend_by_genre():
    weighted_avg_ratings = ratings.groupby('movie_id').apply(weighted_average)
    weighted_avges = pd.DataFrame({'movie_id': weighted_avg_ratings.index, 'WeightedAvgRating': weighted_avg_ratings.values})
    
    
    avg_ratings = ratings.groupby('movie_id').apply(regular_average)
    avges = pd.DataFrame({'movie_id': avg_ratings.index, 'AvgRating': avg_ratings.values})
    
    num_ratings = ratings.groupby('movie_id')['rating'].count()
    num_ratings = pd.DataFrame({'movie_id':num_ratings.index, 'NumRating': num_ratings.values})
    
    movies_with_weighted_ratings = pd.merge(pd.merge(pd.merge(movies, weighted_avges, on='movie_id'), avges, on='movie_id'), num_ratings, on='movie_id').sort_values(by='WeightedAvgRating', ascending=False)

    genre_expanded = movies_with_weighted_ratings['genres'].str.get_dummies('|')
    movie_genre_rating = pd.concat([movies_with_weighted_ratings[["movie_id","title","WeightedAvgRating","AvgRating","NumRating"]], genre_expanded], axis=1)
    return movie_genre_rating

# returns the top n most highest rated movies in a given genre.
def get_recommendation_by_genre(genre, data, n=5):
    genre_movies = data[data[genre] == 1]
    recommended_movies = genre_movies.sort_values(by='WeightedAvgRating', ascending=False).head(n)
    return recommended_movies[['movie_id', 'AvgRating', 'title']].sort_values(by='AvgRating', ascending=False)


users, movies, ratings = read_data()
movie_genre_rating = setup_recommend_by_genre()

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

# Save the top movie recommendations for each genre to avoid recomputing them each time.
top_movies = []
for genre in genres:
    # Get the top 10 movies in that genre
    recommended_movies = get_recommendation_by_genre(genre, movie_genre_rating, n=10)
    # Add the genre to the DataFrame
    recommended_movies['genre'] = genre
    # Append the DataFrame to the list
    top_movies.append(recommended_movies)
# Concatenate the list into a single DataFrame
top_movies_df = pd.concat(top_movies)

def myIBCF(newuser): #newuser is a np array where unreviewed movies are nan
    S = pd.read_csv("https://raw.githubusercontent.com/briany8/cs598/main/S_top30.csv")
    #S = pd.read_csv("https://raw.githubusercontent.com/SophiaY404/CS598-PSL-MovieRecommendApp/main/S0.csv")
    S.index = S.columns
    newuser = pd.Series(newuser).reindex(S.columns)
    w = pd.DataFrame(data=newuser.values, index=S.columns)

    out = np.zeros(3706)
    for j in range(3706):
        l = S.columns[j]
        Sl = S.loc[l].dropna().index
        rhs_total = 0
        lhs_denom_total = 0
        for k in range(len(Sl)):
            i = Sl[k]
            Sli = S.loc[l, i]
            wi = w.loc[i][0]
            if (not np.isnan(wi)):
                rhs_total = rhs_total + (Sli * wi)
                lhs_denom_total = lhs_denom_total + Sli
        if (lhs_denom_total == 0):
            out[j] = 0
        else: 
            out[j] = rhs_total / lhs_denom_total
    out = out+1
    df = pd.DataFrame(data = out, index = S.columns)
    rated = w.dropna().index
    top10_without_rated = df.drop(rated).nlargest(10, 0).reset_index()
    top10_without_rated = top10_without_rated[top10_without_rated.columns[0]]  #keep only the movie_id column
    return top10_without_rated.to_frame(name='movie_id')


def get_displayed_movies():
    return movies.sample(100) #get random sample of movies
    #return movies.head(100)

def get_recommended_movies(new_user_ratings):
    recs = myIBCF(new_user_ratings)
    recs['movie_id'] = recs['movie_id'].str[1:].astype(int) #remove 'm' in front of movie id
    movie_ids = recs['movie_id'].tolist()
    return movies.set_index('movie_id').loc[movie_ids].reset_index()

def get_popular_movies(genre: str):
    if genre == genres[1]:
        return movies.head(10)
    else: 
        return top_movies_df[top_movies_df['genre'] == genre]
