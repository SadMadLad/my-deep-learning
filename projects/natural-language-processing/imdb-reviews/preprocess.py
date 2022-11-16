import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))


def TextsAndLabels():
    directory = os.path.join(base_dir, "dataset")
    movie_names = os.listdir(directory)

    texts = []
    labels = []

    for movie in movie_names:
        movie_directory = os.path.join(directory, movie, "movieReviews.csv")
        movie_reviews = pd.read_csv(movie_directory)

        texts.append(movie_reviews['Review'])
        labels.append(movie_reviews["User's Rating out of 10"])

    return [*texts], [*labels]


train_texts, train_labels = TextsAndLabels()
for text in train_texts:
    print(text)
