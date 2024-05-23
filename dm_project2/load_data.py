import os
import pandas as pd


class MovieLensDataset:

    name: str
    links: pd.DataFrame
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame

    def __init__(self, dataset_name: str = 'ml-latest-small') -> None:
        dirname = os.path.dirname(__file__)
        self.name = dataset_name
        self.links = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/links.csv'))
        self.movies = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/movies.csv'))
        self.ratings = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/ratings.csv'))
        self.tags = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/tags.csv'))
    
    def to_single_dataframe(self) -> pd.DataFrame:
        return self.ratings.merge(self.movies, on='movieId').merge(self.tags.groupby('movieId')['tag'].apply(lambda x: '|'.join(set(x))), on='movieId')


if __name__ == '__main__':
    print(MovieLensDataset().to_single_dataframe().head())
