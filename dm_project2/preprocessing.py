import pandas as pd
from dataset import MovieLensDataset
from sklearn.base import TransformerMixin, BaseEstimator
    

class MovieLensDatasetPreprocessor(BaseEstimator, TransformerMixin):

    _dataset: MovieLensDataset

    def __init__(self) -> None:
        self._dataset = None

    def fit(self, dataset: MovieLensDataset) -> None:
        self._dataset = dataset
    
    def transform(self, dataset: MovieLensDataset) -> "MovieLensDatasetPreprocessor":
        return self
    
    def fit_transform(self, dataset: MovieLensDataset) -> "MovieLensDatasetPreprocessor":
        self.fit(dataset)
        return self.transform(dataset)
    
    def movies_ohe(self) -> pd.DataFrame:
        return self._dataset.get_movies()["genres"].str.get_dummies("|").drop(columns="(no genres listed)")    
    
    def preprocess_ratings(self) -> pd.DataFrame:
        return pd.get_dummies(self._dataset.get_ratings().join(self.movies_ohe()).drop('timestamp', axis=1).droplevel('movieId'), columns=['rating']).astype(bool)
    
    def get_dataset(self) -> MovieLensDataset:
        return self._dataset
