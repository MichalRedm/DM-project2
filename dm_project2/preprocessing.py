import pandas as pd
from dataset import MovieLensDataset
from sklearn.preprocessing import StandardScaler


class Preprocessing:

    _dataset: MovieLensDataset

    def __init__(self, dataset: MovieLensDataset) -> None:
        self._dataset = dataset

    def preprocess_movies(self) -> pd.DataFrame:
        movies = self._dataset.get_movies()
        movies_ohe = movies["genres"].str.get_dummies("|").drop(columns="(no genres listed)")
        return pd.DataFrame(StandardScaler().fit_transform(movies_ohe), columns=movies_ohe.columns)


def to_single_dataframe(dataset: MovieLensDataset) -> pd.DataFrame:
    return dataset.get_ratings().merge(dataset.get_movies(), on='movieId').merge(dataset.get_tags().groupby('movieId')['tag'].apply(lambda x: '|'.join(set(x))), on='movieId')
