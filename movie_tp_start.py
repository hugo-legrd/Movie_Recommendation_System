#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:22:20 2020

@author: smalinow
"""
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


unames = ['movie_id', 'Title', 'genres']
movies = pd.read_csv('./movies.csv', header=None, names = unames, sep = '::')

ratings = pd.read_csv('./ratings_train.csv', index_col=0)
ratings_test = pd.read_csv('./ratings_test.csv', index_col=0)
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_csv('users.csv', sep = '::', header=None, names = unames)


########################### Question 1 #################################
def create_new_column_year(x):
    return x.split('(')[-1].replace(')', '')

def delete_string_inside_parenthesis(x):
    return x.split(' (')[0]


movies['Year'] = movies['Title'].apply(create_new_column_year)
movies['Title'] = movies['Title'].apply(delete_string_inside_parenthesis)



########################### Question 2 #################################
"""Find the movie release in 1995 that has the best ratings"""
movies_1995 = movies[movies['Year'] == '1995']

"""Now we have to merge the movies_1995 with the ratings"""
movies_1995_ratings = pd.merge(movies_1995, ratings, on='movie_id')

"""Now we have to group by the title and calculate the mean of the ratings"""
movies_1995_ratings_mean = movies_1995_ratings.groupby('Title')['rating'].mean()


"""Now we have to sort the values"""
movies_1995_ratings_mean.sort_values(ascending=False, inplace=True)

"""Now we have to plot the top 10 movies"""
#movies_1995_ratings_mean.head(10).plot(kind='barh')

"""movie_id of Gate Of Heavenly peace"""
movies[movies['Title'] == 'Gate of Heavenly Peace, The ']

"""Number of Ratings """
"""and how many people have rated this movie"""
movies_1995_ratings[movies_1995_ratings['Title'] == movies_1995_ratings_mean.head(1).index[0]]['rating'].count()

########################### Question 3 #################################
"""Count the number of ratings for each movies in 1995"""
movies_1995_ratings_count = movies_1995_ratings.groupby('Title')['rating'].count()
more_than_10_ratings = movies_1995_ratings_count[movies_1995_ratings_count > 10]

"""Get the mean of the ratings for the movies with more than 10 ratings"""
more_than_10_ratings_mean = movies_1995_ratings_mean[movies_1995_ratings_mean.index.isin(more_than_10_ratings.index)]
more_than_10_ratings_mean.sort_values(ascending=False)
#more_than_10_ratings_mean.head(10).plot(kind='barh')

########################### Question 4 #################################
"""Create the bins and cut movie['Year']"""
movies['Year'] = pd.to_numeric(movies['Year'])
movies['Year'] = pd.cut(movies['Year'], bins=[0, 1940, 1959, 1979, 2020], labels=['<1940', '[1940,1960[', '[1960, 1980[', '>1980'])
"""Merge the two dataframe to get back the ratings"""
movies_and_ratings = pd.merge(movies, ratings, on='movie_id')

"""Count the number of ratings by bins"""
ratings_by_bins = movies_and_ratings.groupby('Year')['rating'].count()
"""Get the number of unique films by bins"""
movies_by_bins = movies_and_ratings.groupby('Year')['movie_id'].nunique()
"""Compute the average number of ratings by movies"""
avgNumber_ratings_by_bins = ratings_by_bins/movies_by_bins

########################### Question 5 #################################
"""Here is the average rating in the dataset"""
average_rating = ratings['rating'].mean()

########################### Question 6 #################################
"""Here is the average rating given by user 148"""
average_rating_user_148 = ratings[ratings['user_id'] == 148]['rating'].mean()

########################### Question 7 #################################
"""Here is the distribution of ratings for the movie ’Toy Story’, so that we can see if it is a good movie or not."""
id_toy_story = movies[movies['Title'] == 'Toy Story']['movie_id'][0]
toy_story_ratings = ratings[ratings['movie_id'] == id_toy_story]['rating']
toy_story_ratings_count = toy_story_ratings.value_counts()

########################### Question 8 #################################
"""Merge the two dataframe to get back the users"""
ratings_and_users = pd.merge(ratings, users, on='user_id')
"""Count the number of ratings by group"""
ratings_by_occupation = ratings_and_users.groupby(['occupation', 'gender'])['rating'].count()
"""Get the number of unique films by group"""
movies_by_occupation = ratings_and_users.groupby(['occupation', 'gender'])['movie_id'].nunique()
"""Compute the average number of ratings by movies"""
avgNumber_ratings_by_occupation = ratings_by_occupation/movies_by_occupation

########################### Question 9 #################################
ratings_by_age = ratings_and_users.groupby('age')['rating'].mean()
#We don't really see a huge difference on the average of ratings between age, maybe 56+ are less harsh and [18-25] are more difficult to please
########################### Question 10 #################################
ratings_and_users_and_movies = pd.merge(ratings_and_users, movies, on='movie_id')
ratings_by_gender_and_genres = ratings_and_users_and_movies.groupby(['genres', 'gender'])['rating'].mean()
#Drama|Romance|Western (4(F) vs 3(M)), Crime|Mistery (2.5(F) vs 3.517(M))


######################################### 1.2 #########################################
def predict_a_rating_regarding_a_movie(m) :
    #Here we predict the rating of a movie m for a user u
    #We get the movie id of m
    #movie_id = movies[movies['Title'] == m]['movie_id']
    #We get the ratings of the movie
    movie_ratings = ratings[ratings['movie_id'] == m]['rating']
    #We get the average rating of the movie
    average_rating_movie = movie_ratings.mean()
    return average_rating_movie

predict_a_rating_regarding_a_movie(1)

ratings_test['predicted_rating_V1'] = ratings_test['movie_id'].apply(predict_a_rating_regarding_a_movie)

pred = ratings_test['predicted_rating_V1']

def evaluation_function(p):
    res = np.array([1, 3, 2, 5, 3, 4, 4, 3, 3, 5, 5, 5, 4, 2, 4, 2, 5, 5, 4, 4, 5, 3, 5, 5, 1, 3, 2, 4, 3, 4, 2, 3, 3, 3, 5, 5, 3, 4, 4, 4, 5, 5, 3, 3, 5, 3, 5, 2, 2, 3, 4])
    return mean_squared_error(p,res)

# Question : what is the prediction error made by this first reccomender system? Remember this value, you’ll have to compare it with errors of other srategies (in the next sections).
evaluation_function(pred)

#1.113208


######################################### 1.3 #########################################
"""Which movie has the highest rating """
movies_ratings_mean = movies_and_ratings.groupby('Title')['rating'].mean()
"""Now we have to sort the values"""
movies_ratings_mean.sort_values(ascending=False, inplace=True)

"""Now we have to plot the top 13 movies"""
#movies_ratings_mean.head(13).plot(kind='barh')
#13 movies has 5 star of average ratings
top13_movies = movies_ratings_mean.head(13)

"""How many ratings have each of the top13_movies"""
top13_movies_ratings_count = movies_and_ratings[movies_and_ratings['Title'].isin(top13_movies.index)].groupby('Title')['rating'].count()
top13_movies_ratings_count.sort_values(ascending=False, inplace=True)

# write a function that computes the weighted rating of a movie given its id movie_id and a rating matrix (as the one you have, with all the ratings given by users to the different movies)
def weighted_rating(movie_id, rating_matrix):
    """Here we compute the weighted rating of a movie given its id movie_id and a rating matrix"""
    # get the number of ratings for the movie
    v = rating_matrix[rating_matrix['movie_id'] == movie_id]['rating'].count()
    # get the mean of ratings for the movie
    R = rating_matrix[rating_matrix['movie_id'] == movie_id]['rating'].mean()
    # get the mean of ratings for all movies
    C = rating_matrix['rating'].mean()
    # get the 75-percentile value of number of ratings for the different movies of the dataset
    n = rating_matrix['movie_id'].value_counts().quantile(0.75)

    # compute the weighted rating
    weighted_rating = (v/(v+n)) * R + (n/(v+n)) * C
    return weighted_rating

#what is the movie with the highest weighted average rating ?
#movies['weighted_rating'] = movies['movie_id'].apply(weighted_rating, rating_matrix=ratings)
#Sort and show the 10 best weighted rating movies and their name
#movies.sort_values(by='weighted_rating', ascending=False).head(10)[['Title', 'weighted_rating']]


#use this weighted rating to predict the ratings of all pairs in the test dataset. (exactly the same
#as before, but the average rating of movies is replaced by the weighted average).
"""En commentaire car trop long sinon"""
#ratings_test['predicted_rating_V2'] = ratings_test['movie_id'].apply(weighted_rating, rating_matrix=ratings)

#predV2 = ratings_test['predicted_rating_V2']

#evaluation_function(predV2)
#1.06505

######################################### 1.4 #########################################
# To predict the rating given by user u to movie m, we have to compute the average rating of movie m but only ratings given by users of the same gender as user u.
def get_the_gender(user_id) :
    return users[users['user_id'] == user_id]['gender'].values[0]


def predict_a_rating_regarding_gender(gender, movie_id) :
    return ratings_and_users[(ratings_and_users['movie_id'] == movie_id) & (ratings_and_users['gender'] == gender)]['rating'].mean()

#ratings_test["prediction_V3"] = ratings_test.apply(lambda x: predict_a_rating_regarding_gender(get_the_gender(x['user_id']), x['movie_id']), axis=1)
#predV3 = ratings_test['prediction_V3']

#print(evaluation_function(predV3))

######################################### 1.5 #########################################
##################################### Question 1 ######################################
def similarity_between_two_movies(m1, m2) : 
    similarity = 0

    #Si m1 OU m2 à - de 40 notes alors similarity = -1
    if(ratings[ratings['movie_id'] == m1]['rating'].count() < 40 or ratings[ratings['movie_id'] == m2]['rating'].count() < 40) : 
        similarity = -1
    else : 
        #On regarde quels sont les users qui ont notés les 2 : sous la forme 'user_id' -> False/True
        users_rated_both_movies_for_m1 = ratings[ratings['movie_id']== m1]['user_id'].isin(ratings[ratings['movie_id']==m2]['user_id'])
        users_rated_both_movies_for_m2 = ratings[ratings['movie_id']== m2]['user_id'].isin(ratings[ratings['movie_id']==m1]['user_id'])

        #On récupère les ID où c'est True, donc la c'est les user_id qui ont notés les 2 films
        usersId_rated_both_movies_for_m1 = ratings[ratings['movie_id']== m1][users_rated_both_movies_for_m1]['user_id']
        usersId_rated_both_movies_for_m2 = ratings[ratings['movie_id']== m2][users_rated_both_movies_for_m2]['user_id']

        #On fait une Series avec les notes de m1
        ratings_m1 = ratings[ratings['movie_id']== m1]['rating']
        #On récupère les notes de m1 pour les users qui ont notés les 2 films
        ratings_m1_by_user = ratings_m1[ratings_m1.index.isin(usersId_rated_both_movies_for_m1.index)]
        #On fait une Series avec les notes de m2
        ratings_m2 = ratings[ratings['movie_id']== m2]['rating']
        #Pareil 
        ratings_m2_by_user = ratings_m2[ratings_m2.index.isin(usersId_rated_both_movies_for_m2.index)]

        #Faire une matrice de corrélation avec les ratings de m1 et m2 pour comparer les 2
        res = np.corrcoef(ratings_m1_by_user, ratings_m2_by_user)

        #Récupérer la valeur qui nous intéresse
        similarity = res[0,1]
    return similarity


#print(similarity_between_two_movies('Toy Story', 'Toy Story 2'))


##################################### Question 2 ######################################
def predict_a_movie_nearest_neighbour(m, u, k) : 
    #Get all the movies rated by u
    movies_rated_by_u = ratings[ratings['user_id'] == u]['movie_id']
    #Now get the titles and movie_id of the movies rated by u
    movies_rated_by_u_titles = movies[movies['movie_id'].isin(movies_rated_by_u)]['Title']
    #Compute the similarity between m and all the movies rated by u using similarity_between_two_movies
    similarity = movies_rated_by_u_titles.apply(similarity_between_two_movies, m2=m)
    #Get the k movies with the highest similarity and get the movie Title
    k_most_similar_movies = similarity.sort_values(ascending=False).head(k)
    similar_movies_id = movies.loc[k_most_similar_movies.index]['movie_id']
    #Get the ratings of the k_most_similar_movies
    k_most_similar_movies_ratings = ratings[ratings['movie_id'].isin(similar_movies_id)]
    #Print the k_most_similar_movies_ratings with user_id == u_id
    user_k_most_similar_movies_rating = k_most_similar_movies_ratings[k_most_similar_movies_ratings['user_id'] == u]
    #Predict rating(m, u) as : (1/k) * sum(rating(m', u) for m' in k_most_similar_movies)
    average_rating = (1/k) * user_k_most_similar_movies_rating['rating'].sum()
    return average_rating


print(predict_a_movie_nearest_neighbour(1, 1, 5))


##################################### Question 3 ######################################
"""En commentaire car l'éxécution prend trop de temps sinon"""
#ratings_test['predicted_rating_V3_k1'] = ratings_test.apply(lambda x : predict_a_movie_nearest_neighbour(x['movie_id'], x['user_id'], 1), axis=1)
#predV3_k1 = ratings_test['predicted_rating_V3_k1']

#evaluation_function(predV3_k1)
#1.9215686274509804"""


##################################### Question 4 ######################################
"""En commentaire car l'éxécution prend trop de temps sinon"""
#ratings_test['predicted_rating_V3_k3'] = ratings_test.apply(lambda x : predict_a_movie_nearest_neighbour(x['movie_id'], x['user_id'], 3), axis=1)
#predV3_k3 = ratings_test['predicted_rating_V3_k3']

#evaluation_function(predV3_k3)
#1.264705882352941


######################################### 1.6 #########################################
##################################### Question 1 ######################################
def similarity_between_user(u, v) :
    distance = 0
    movies_rated_by_u = ratings[ratings['user_id'] == u]['movie_id']
    movies_rated_by_v = ratings[ratings['user_id'] == v]['movie_id']
    movies_rated_by_both = movies_rated_by_u[movies_rated_by_u.isin(movies_rated_by_v)]
    if movies_rated_by_both.count() < 20 : 
        distance = 10000
    else :
        distance = (1/movies_rated_by_both.count()) * (np.sum(np.square((ratings[ratings['movie_id'].isin(movies_rated_by_both) & (ratings['user_id'] == u)]['rating']).values - (ratings[ratings['movie_id'].isin(movies_rated_by_both) & (ratings['user_id'] == v)]['rating'])).values))
    return distance

#print(similarity_between_user(5, 8))


##################################### Question 2 ######################################
def predict_a_movie_user_nearest_neighbour(m, u, k) : 
    #Get all the movies rated by u
    movies_rated_by_u = ratings[ratings['user_id'] == u]['movie_id']
    #Now get the titles and movie_id of the movies rated by u
    movies_rated_by_u_titles = movies[movies['movie_id'].isin(movies_rated_by_u)]['Title']
    #Get all the user that have rated m
    users_rated_m = ratings[ratings['movie_id'] == m]['user_id']
    similarity = users_rated_m.apply(similarity_between_user, v=u)
    #Get the k users with the lowest similarity except the first one
    k_most_similar_users = similarity.sort_values(ascending=True).head(k+1).tail(k)
    #Get the user id of the k_most_similar_users
    similar_users_id = ratings.loc[k_most_similar_users.index]['user_id']
    average_rating = (1/k) * (ratings[ratings['user_id'].isin(similar_users_id.values) & (ratings['movie_id'] == m)]['rating'].sum())
    return average_rating

#print(predict_a_movie_user_nearest_neighbour(1, 1, 1))

##################################### Question 3 ######################################
#ratings_test['predicted_rating_V4_k1'] = ratings_test.apply(lambda x : predict_a_movie_user_nearest_neighbour(x['movie_id'], x['user_id'], 1), axis=1)
#predV4_k1 = ratings_test['predicted_rating_V4_k1']

#print(evaluation_function(predV4_k1))
#1.1372549019607843"""

#ratings_test['predicted_rating_V4_k2'] = ratings_test.apply(lambda x : predict_a_movie_user_nearest_neighbour(x['movie_id'], x['user_id'], 2), axis=1)
#predV4_k2 = ratings_test['predicted_rating_V4_k2']

#print(evaluation_function(predV4_k2))
#0.9705882352941176


#K3 = 0.9368191721132899

#%%