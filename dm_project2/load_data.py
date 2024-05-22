import os
import pandas as pd


def load_data(dataset: str = 'ml-latest-small') -> pd.DataFrame:

    dirname = os.path.dirname(__file__)

    links = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset}/links.csv'))
    movies = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset}/movies.csv'))
    ratings = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset}/ratings.csv'))
    tags = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset}/tags.csv'))

    return ratings
