#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:53:56 2020

@author: smalinow
"""

"""
Updated on Wed Nov 23 11:30:00 2022

@author: jesterbet
"""
#%%
import pandas as pd
import numpy as np
from scipy.spatial import distance
#from functions import *

metadata = pd.read_csv('metadata.csv', low_memory=False)
metadata.iloc[0]
metadata.overview[0]

# metadata is a dataframe that contains information about 5000 movies including an overview (synopsis) of each movie
# The idea of this section is to compare movies by looking only at their overviews
# It is a langage processing problem : how to compare two written sentences ?
# I will explain you how to proceed:
# We need to transform a sentence (composed of words) into a numerical vector.
# For this, we first need a list of all possible words (i will give you one right after)
# For instance let's imagine that the only possible words are 
# ['cat', 'drinks', 'dog', 'mouse', 'plays', 'milk']
# And we want to work with the sentence 'A cat drinks some milk'.
# The following operations need to be done : 
# - Transform the sentence into a list of words:
# Here : ['a', 'cat', 'drinks', 'some', 'milk']
# - Remove from this list words that are called stopwords (common, not very informative)
# I will give you a list of stopwords after
# The words 'a', 'some' are stopwords.
# So the list of words of the sentence now is : ['cat', 'drinks', 'milk']
#  - the sentence is then transformed into a numerical vector that represents
# the frequency of occurence of each possible words inside the sentence
# Here we have
# - 1 occurence of 'cat'
# - 1 occurence of 'drinks'
# - 0 occurence of 'dog'
# - 0 occurence of 'mouse'
# - 0 occurence of 'plays'
# - 1 occurence of 'milk'
# So the numerical vector that represents this sentence is : 
# [1 1 0 0 0 1] that needs to be normalized so that the sum is 1 (it represents the probability of
# words in the sentence)
# So [0.333 0.333 0 0 0 0.333] is the final representation of the sentence 'A cat drinks some milk'
# Similarly, the sentence 'A dog plays with a cat and a mouse' is represented as:
# [0.25 0 0.25 0.25 0.25 0]
# To compare 2 sentences represented by such vectors, a distance is well adapted : 
# the cosine distance (you can look on the web if interested)
# but basically you can use it easily :
# distance.cosine([0.333,0.333,0,0,0,0.333], [0.25,0,0.25,0.25,0.25,0])
# Here it will give you 0.7113
# if the distance is 0, it means that the 2 vectors are the same. the higher the distance
# the more different are the vectors (and hence the sentences)

# You will apply this technique to compare the synopsis of your movies. For that, you will need to:
#  - transform every synopsis into a list of words (i will give you this function)
#  - remove stop words from the list of words of each movie 
#  - compute the occurence frequency vector using a list of all possible words (that I'll give you)
# And then you can compare 2 movies by the cosine distance between their numerical representation


# Here you will load the list of all words in our dataset (all the words of all the movies synopsis)
unique_words = pd.read_csv('unique_words.csv', index_col=0)
unique_words = unique_words.iloc[:,0].to_list()
# The list unique_words contains all the words that appear in the overviews of the movies in metadata (only one occurence of each word)
# Question : how many different words ? 
"""len(unique_words)  = 24240"""
# Have a look at some of these words


# Here you will load the list of stop words
stop_words =pd.read_csv('stopwords.csv', index_col=0)
stop_words = stop_words.iloc[:,0]
stop_words.dropna(inplace=True)
stop_words = stop_words.to_list()
# The variable stop_words containsis a list of all the english words that are considered as stopwords, i.e. words that are not interesting to compare sentences 
# Have a look at some words of this list
# Question : Can you find the word 'and' inside the list of stopwords ? 
"""stop_words.index('and') = 63"""


# The function below create a list of words from the sypnosis of the i^th movie of metadata
def create_list_words(i, metadata):
    if(type(metadata.overview[i])!=str):
        return []
    else : 
        a = str.split(metadata.overview[i])
    a = [a[i].replace('.','') for i in np.arange(len(a))]
    a = [a[i].replace(',','') for i in np.arange(len(a))]
    a = [a[i].replace("'s",'') for i in np.arange(len(a))]
    a = [a[i].replace("'",'') for i in np.arange(len(a))]
    a = [a[i].replace('_','') for i in np.arange(len(a))]
    a = [a[i].replace('-','') for i in np.arange(len(a))]
    a = [a[i].replace('"','') for i in np.arange(len(a))]
    a = [a[i].replace('(','') for i in np.arange(len(a))]
    a = [a[i].replace(')','') for i in np.arange(len(a))]
    a = [a[i].replace(':','') for i in np.arange(len(a))]
    """Remove the stop words"""
    a = [i for i in a if i not in stop_words]
    a = [j.lower() for j in a]
    return a

#If you want the list of words of the first movie: 
a = create_list_words(0, metadata)
# print(a)
# Question : Can you find the word 'and' inside this list ? 
"""a.index('and') = 36"""
# You should. As 'and' is a stop word, we need to remove it from the list
# Question : modify the function 'create_list_words' so that the returned list do not contain any stop words. 
# So you should remove all the stop words from 'a' before returning it.


# Question : Now create a function 'word_frequencies (i, metadata, unique_words)'
# that computes the occurence frequency vector of the words in the synopsis of movie i
# the results should be a Series of length equal to the number of possible words
# and the elements of the result shoud be the occurence frequency of the possible words
# (as [0.333 0.333 0 0 0 0.333] in the toy example below)
def word_frequencies(i, metadata, unique_words) : 
    a = create_list_words(i, metadata)
    # Count the occurence of each word in a
    b = pd.Series(a, dtype='string').value_counts()
    # Now transform b into an nparray and each elements of the array is the occurence frequency of the corresponding word
    c = np.zeros(len(unique_words))
    for j in np.arange(len(unique_words)) :
        if unique_words[j] in b.index :
            c[j] = b[unique_words[j]]/len(a)
    return c


# Question : apply your function 'word_frequencies' to the first movie of metadata
occurence_first_movie = word_frequencies(0, metadata, unique_words)
# Question : apply your function 'word_frequencies' to the second movie of metadata
occurence_second_movie = word_frequencies(1, metadata, unique_words)
# Compute the cosine distance between these 2 movies
distance_first_second = distance.cosine(occurence_first_movie,occurence_second_movie)


# Then, write a function 'my_cosine(metadata, i, j) that computes the cosine distance
# between movie i and j of metadata
def my_cosine(metadata, i, j) :
    occurence_movie_i = word_frequencies(i, metadata, unique_words)
    occurence_movie_j = word_frequencies(j, metadata, unique_words)
    if occurence_movie_i.sum() == 0 or occurence_movie_j.sum() == 0 :
        return 10
    return distance.cosine(occurence_movie_i,occurence_movie_j)


# Create a DataFrame that contains the word frequencies representations of all the movies
# in your dataset (about 5000). These representations shoudl be stored in the rows of the DataFrame
# (might take a long time to compute, be patient)
def frequencies_all_movies(metadata, unique_words) :
    frequ_all_movies = pd.DataFrame(np.zeros((len(metadata),len(unique_words))))
    for i in range(len(metadata)) :
        frequ_all_movies.iloc[i,:] = word_frequencies(i, metadata, unique_words)
    return frequ_all_movies
# freq_all_movies = frequencies_all_movies(metadata, unique_words)

# Question : Can you find the 4 movies that are the most similar to the first movie of metadata, using the DataFrame frequ_all_movies and the function 'my_cosine' ? Also make sur that overviews are not empty.
# def most_similar(i):
#     #Compute the cosine distance between movie i and all the other movies
#     dist = np.zeros(frequ_all_movies.shape[0])
#     freq_i = frequ_all_movies.iloc[i,:]
#     for j in range(len(dist)) :
#         freq_j = frequ_all_movies.iloc[j,:]
#         dist[j] = my_cosine(metadata, freq_i, freq_j)
#     #Find the 4 movies that are the closest to movie i
#     index = np.argsort(dist)
#     return index[1:5]

def most_similar(i) :
    similar_movies = []
    for j in range(len(metadata)):
        cosine_dist = my_cosine(metadata, i, j)
        similar_movies.append((metadata.title[j], cosine_dist))
    similar_movies.sort(key=lambda x: x[1])
    return similar_movies[1:5]

# Try your function to find the 4 most similar movies to:
# - 'Toy Story'
# - 'Dr. No' (a James bond)
# - 'The Shawshank Redemption' (a movie about a prison break)

# Find a movie in the dataset using its title
def find_movie(title):
    for i in np.arange(len(metadata)):
        if(metadata.title[i]==title):
            return i
    return -1

print("'Toy Story' similar movies : ", most_similar(0))
# ('Toy Story 2', 0.5957739582727783), ('Man on the Moon', 0.8514778685534988), ('Condorman', 0.8747551417829701), ('Rebel Without a Cause', 0.8774754926475492)]
# print("'Dr. No' similar movies : ", most_similar(find_movie("Dr. No")))
# [('Live and Let Die', 0.7609542781331213), ('From Russia with Love', 0.7741230242736872), ('GoldenEye', 0.800795231777601), ('The Man with the Golden Gun', 0.8237731557874397)]
# print("'The Shawshank Redemption' similar movies : ", most_similar(find_movie("The Shawshank Redemption")))
# [('A Further Gesture', 0.8101858494086758), ('Brubaker', 0.8174258141649446), ('Penitentiary', 0.8227706107603583), ('Cool Hand Luke', 0.8278674068352259)]
# %%
