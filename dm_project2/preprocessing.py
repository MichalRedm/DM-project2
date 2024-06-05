import pandas as pd
from dataset import MovieLensDataset
from sklearn.base import TransformerMixin, BaseEstimator
    

class MovieLensDatasetPreprocessor(BaseEstimator, TransformerMixin):
    """
    Class used for performing basic preprocessing of
    the Movie Lens dataset.
    """

    _dataset: MovieLensDataset

    def __init__(self) -> None:
        self._dataset = None

    def fit(self, dataset: MovieLensDataset) -> None:
        """Fits the data preprocessor to the dataset."""
        assert isinstance(dataset, MovieLensDataset), 'Dataset preprocessed by MovieLensDatasetPreprocessor should be of type MovieLensDataset.'
        self._dataset = dataset
    
    def transform(self, dataset: MovieLensDataset) -> "MovieLensDatasetPreprocessor":
        """Transform the dataset."""
        assert isinstance(dataset, MovieLensDataset), 'Dataset preprocessed by MovieLensDatasetPreprocessor should be of type MovieLensDataset.'
        return self
    
    def fit_transform(self, dataset: MovieLensDataset) -> "MovieLensDatasetPreprocessor":
        """Combines fitting to the dataset and transforming it."""
        self.fit(dataset)
        return self.transform(dataset)
    
    def movies_ohe(self) -> pd.DataFrame:
        """
        Perfroms one-hot encoding of the movie genres.

        Returns
        -------
        pandas.DataFrame
            One-hot encoded movies.
        """
        return self._dataset.get_movies()["genres"].str.get_dummies("|").drop(columns="(no genres listed)")    
    
    def preprocess_ratings(self) -> pd.DataFrame:
        """
        Performs ratings preprocessing.

        Returns
        -------
        pandas.DataFrame
            Preprocessed ratings.
        """
        return pd.get_dummies(self._dataset.get_ratings().join(self.movies_ohe()).drop('timestamp', axis=1).droplevel('movieId'), columns=['rating']).astype(bool)
    
    def get_dataset(self) -> MovieLensDataset:
        """
        Provides the Movie Lens dataset on which the
        preprocessing is performed.

        Returns
        -------
        MovieLensDataset
            Dataset.
        """
        return self._dataset
