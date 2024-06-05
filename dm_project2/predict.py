import sys
import warnings
import pandas as pd
from dataset import MovieLensDataset
from preprocessing import MovieLensDatasetPreprocessor
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List


class Predictor:
    """
    Rating predictor for the Movie Lens dataset
    based on association rules.
    """

    _preprocessor: MovieLensDatasetPreprocessor
    _ratings_preprocessed: pd.DataFrame

    def __init__(self) -> None:
        self._preprocessor = None

    def fit(self, preprocessor: MovieLensDatasetPreprocessor) -> None:
        """Fits the predictor to the preprocessed data."""
        if not isinstance(preprocessor, MovieLensDatasetPreprocessor):
            raise ValueError(f'Parameter of fit method should be of type MovieLensDatasetPreprocessor.')
        self._preprocessor = preprocessor
        self._ratings_preprocessed = preprocessor.preprocess_ratings()
    
    def _get_avg_movie_rating(self, movie_id: int) -> float:
        """
        Computes the average rating of the movie.

        Parameters
        ----------
        movie_id : int
            Id of the movie in the Movie Lens dataset.
        
        Returns
        -------
        float
            Computed average rating.
        """
        ratings = self._preprocessor.get_dataset().get_ratings()
        movie_avg = ratings[ratings.index.get_level_values('movieId') == movie_id]['rating'].mean()
        return movie_avg if not pd.isna(movie_avg) else 3.5
    
    def _get_avg_user_rating(self, user_id: int) -> float:
        """
        Computes the average rating given by a specific user.

        Parameters
        ----------
        user_id : int
            Id of the user in the Movie Lens dataset.
        
        Returns
        -------
        float
            Computed average rating.
        """
        ratings = self._preprocessor.get_dataset().get_ratings()
        return ratings[ratings.index.get_level_values('userId') == user_id]['rating'].mean()
    
    def predict(
        self,
        user_id: int,
        movie_id: int,
        threshold_itemsets: float = 0.01,
        threshold_rules: float = 0.01,
        weighted_mean_metric: str = 'confidence',
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> float:
        """
        Predicts rating for a given user-movie pair.

        Parameters
        ----------
        user_id : int
            Id of the user in the Movie Lens dataset.

        movie_id : int
            Id of the movie in the Movie Lens dataset.
        
        threshold_itemsets : float
            Minimum support used in the apriori algorithm.
        
        threshold_rules : float
            Threshold used when generating association rules.

        weighted_mean_metric : str
            Association rule metric that should be used as weights of
            a weighted mean when computing the prediction based on
            association rules. The following metrics are supported:
            - `antecedent support`,
            - `consequent support`,
            - `support`,
            - `confidence`,
            - `lift`.
        
        alpha : float
            Importance of the prediction based on average movie rating
            vs average user rating in the final prediction.
        
        beta : float
            How important is the prediction purely based on association
            rules in the final prediction.
        
        Returns
        -------
        float
            Prediction of the rating (not rounded).
        """
        
        assert isinstance(user_id, int) and user_id > 0, 'user_id should be a positive integer.'
        assert isinstance(movie_id, int) and movie_id > 0, 'movie_id should be a positive integer.'
        assert isinstance(threshold_itemsets, float) and threshold_itemsets >= 0, 'threshold_itemsets should be a nonnegative float.'
        assert isinstance(threshold_rules, float) and threshold_rules >= 0, 'threshold_rules should be a nonnegative float.'
        assert weighted_mean_metric in ('antecedent support', 'consequent support', 'support', 'confidence', 'lift'), 'Unknown weighted mean metric.'
        assert isinstance(alpha, float) and 0 <= alpha <= 1, 'alpha should be a float from the interval [0, 1].'
        assert isinstance(beta, float) and 0 <= beta <= 1, 'beta should be a float from the interval [0, 1].'

        warnings.filterwarnings('ignore')

        user_prepr = self._ratings_preprocessed[self._ratings_preprocessed.index == user_id]

        frequent_itemsets = apriori(user_prepr, min_support=threshold_itemsets, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=threshold_rules)
        rules = rules[rules['consequents'].apply(lambda x: len(x) == 1 and 'rating' in list(x)[0])]

        movie = self._preprocessor.get_dataset().get_movie_by_id(movie_id)
        genres = frozenset(movie.genres)

        movie_avg = self._get_avg_movie_rating(movie_id)
        user_avg = self._get_avg_user_rating(user_id)
        avg_prediction = movie_avg * alpha + user_avg * (1 - alpha)

        relevant_rules = rules[rules['antecedents'] <= genres]
        relevant_rules.loc[:, 'consequents'] = relevant_rules['consequents'].apply(lambda x: float(list(x)[0].split('_')[1]))
        rules_prediction = (relevant_rules['consequents'] * relevant_rules[weighted_mean_metric]).sum() / relevant_rules[weighted_mean_metric].sum()

        if pd.isna(rules_prediction):
            rules_prediction = avg_prediction
        
        return rules_prediction * beta + avg_prediction * (1 - beta)


def round_rating(rating: float) -> float:
    """
    Rounds arbitrary rating to a proper rating
    (to the nearest half).

    Parameters
    ----------
    rating : float
        Arbitrary rating to be rounded.

    Returns
    -------
    float
        Rounded rating.
    """
    return round(rating * 2) / 2


def main(args: List[str]) -> None:
    
    if len(args) != 3:
        print("Invalid number of arguments.")
        sys.exit(1)
    
    user_id = int(args[0])
    movie_id = int(args[1])
    dataset_name = args[2]

    dataset = MovieLensDataset(dataset_name)
    dataset.delete_rating(user_id, movie_id)
    preprocessor = MovieLensDatasetPreprocessor().fit_transform(dataset)
    predictor = Predictor()
    predictor.fit(preprocessor)
    prediction = predictor.predict(user_id, movie_id)
    
    print(f'Predicted rating for movie "{dataset.get_movie_by_id(movie_id).title}" and user with userId={user_id}: {prediction} (rounded: {round_rating(prediction)})')


if __name__ == "__main__":
    main(sys.argv[1:])
