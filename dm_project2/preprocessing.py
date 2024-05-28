import pandas as pd
from dataset import MovieLensDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Literal, List


class Preprocessing:

    _dataset: MovieLensDataset

    def __init__(self, dataset: MovieLensDataset) -> None:
        self._dataset = dataset

    def movies_ohe(self) -> pd.DataFrame:
        return self._dataset.get_movies()["genres"].str.get_dummies("|").drop(columns="(no genres listed)")    

    def preprocess_movies(self) -> pd.DataFrame:
        movies_ohe = self.movies_ohe()
        return pd.DataFrame(StandardScaler().fit_transform(movies_ohe), columns=movies_ohe.columns)
    
    @staticmethod
    def standarize(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(StandardScaler().fit_transform(df),
                            columns=df.columns, index=df.index)
    
    @staticmethod
    def extract_features(
        df: pd.DataFrame,
        method: Literal['PCA', 'LDA'] = 'PCA',
        n_components: int = 5
    ):
        if method == 'PCA':
            extractor = PCA
        elif method == 'LDA':
            raise NotImplementedError('LDA is not yet implmented.')
        else:
            raise ValueError(f'Unknown feature extraction method: {method}')
        return pd.DataFrame(extractor(n_components).fit_transform(df), 
                         index=df.index)
    
    def preprocess_users(self, movie_cluster: List[int], user_id: int) -> pd.DataFrame:

        ratings = self._dataset.get_ratings().drop('timestamp', axis=1)
        relevant_ratings = ratings[ratings.index.get_level_values('movieId').isin(movie_cluster)]
        mean_rating = relevant_ratings['rating'].mean()

        users = relevant_ratings.join(self.movies_ohe()).apply(
            lambda row: (row * row['rating']).replace(0, mean_rating), axis=1
        ).drop('rating', axis=1).groupby(level=0).mean()
        users.index.name = 'userId'
        users = Preprocessing.standarize(users)
        users = Preprocessing.extract_features(users)

        return users



def to_single_dataframe(dataset: MovieLensDataset) -> pd.DataFrame:
    return dataset.get_ratings().merge(dataset.get_movies(), on='movieId').merge(dataset.get_tags().groupby('movieId')['tag'].apply(lambda x: '|'.join(set(x))), on='movieId')
