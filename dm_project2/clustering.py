import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import Preprocessing
from functools import cache
from typing import List


class Clustering:

    _preprocessing: Preprocessing

    def __init__(self, preprocessing: Preprocessing) -> None:
        self._preprocessing = preprocessing
    
    @cache
    def cluster_movies(self, n_clusters: int = 15) -> pd.DataFrame:
        movies_sd = self._preprocessing.preprocess_movies()
        kmeans = KMeans(n_clusters)
        kmeans.fit(movies_sd)
        return pd.DataFrame({
            'movie_id': movies_sd.index,
            'label': kmeans.labels_
        }).set_index('movie_id')
    
    def get_movie_cluster(self, movie_id: int) -> List[int]:
        movie_clusters = self.cluster_movies()
        label = movie_clusters.loc[movie_id]['label']
        return movie_clusters.index[movie_clusters['label'] == label].to_list()
