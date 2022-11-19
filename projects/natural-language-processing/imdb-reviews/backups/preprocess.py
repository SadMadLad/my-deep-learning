import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))


def TextsAndLabels():
    directory = os.path.join(base_dir, "..","dataset")
    movie_names = os.listdir(directory)

    texts = []
    labels = []

    for movie in movie_names:
        movie_directory = os.path.join(directory, movie, "movieReviews.csv")
        movie_reviews = pd.read_csv(movie_directory)

        texts.append(list(movie_reviews['Review']))
        labels.append(list(movie_reviews["User's Rating out of 10"]))

    texts = [text for review in texts for text in review]
    labels = [label for rating in labels for label in rating]

    valid_ratings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    for i, rating in enumerate(labels):
        if rating not in valid_ratings:
            texts[i] = "|TBD|"
            labels[i] = "|TBD|"
        else:
            labels[i] = int(labels[i])

    labels = list(filter(("|TBD|").__ne__, labels))
    texts = list(filter(("|TBD|").__ne__, texts))

    # print(len(labels))
    # print(len(texts))

    return texts, labels