import pandas as pd
from dataset import MovieLensDataset

def to_single_dataframe(dataset: MovieLensDataset) -> pd.DataFrame:
    return dataset.get_ratings().merge(dataset.get_movies(), on='movieId').merge(dataset.get_tags().groupby('movieId')['tag'].apply(lambda x: '|'.join(set(x))), on='movieId')
