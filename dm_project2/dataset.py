import os
import pandas as pd
from dataclasses import dataclass
from typing import List


class InvalidDatasetException(Exception):
    """Exception thrown when invalid dataset is provided."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidMovieException(Exception):
    """Exception thrown when invalid movie is selected."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidUserException(Exception):
    """Exception thrown when invalid user is selected."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@dataclass
class Movie:
    """Representation of a movie from the Movie Lens dataset."""

    id: int
    """Id of the movie."""

    title: str
    """Title of the movie."""

    genres: List[str]
    """List of genres."""


class MovieLensDataset:
    """
    Common interface for datasets containing 5-star rating and free-text tagging activity
    from [MovieLens](http://movielens.org), a movie recommendation service.

    The dataset comprises of four tables: `links`, `movies`, `ratings` and `tags`.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset. Can be one of the following:
        - `ml-latest-small` (default) - contains 100836 ratings and 3683 tag applications
        across 9742 movies. These data were created by 610 users between March 29, 1996 and
        September 24, 2018. This dataset was generated on September 26, 2018.
        - `ml-latest` - contains 33832162 ratings and 2328315 tag applications across 86537
        movies. These data were created by 330975 users between January 09, 1995 and July 20,
        2023. This dataset was generated on July 20, 2023.
    """

    _name: str
    _links: pd.DataFrame
    _movies: pd.DataFrame
    _ratings: pd.DataFrame
    _tags: pd.DataFrame

    def __init__(self, dataset_name: str = 'ml-latest-small') -> None:
        if dataset_name not in ('ml-latest-small', 'ml-latest'):
            raise InvalidDatasetException(f'Unknown dataset: {dataset_name}')
        dirname = os.path.dirname(__file__)
        self._name = dataset_name
        self._links = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/links.csv')).set_index('movieId')
        self._movies = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/movies.csv')).set_index('movieId')
        self._ratings = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/ratings.csv')).set_index(['userId', 'movieId'])
        self._tags = pd.read_csv(os.path.join(dirname, f'../data/raw/{dataset_name}/tags.csv'))
    
    def get_name(self) -> str:
        """
        Returns
        -------
        str
            Name of the dataset.
        """
        return self._name
    
    def get_links(self) -> pd.DataFrame:
        """
        Provides table containing identifiers that can be used to link to other
        sources of movie data in which each row represents one movie, and
        has the following format:

            `movieId`, `imdbId`, `tmdbId`

        movieId is an identifier for movies used by https://movielens.org.
        E.g., the movie Toy Story has the link https://movielens.org/movies/1.

        imdbId is an identifier for movies used by http://www.imdb.com.
        E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.

        tmdbId is an identifier for movies used by https://www.themoviedb.org.
        E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.

        Use of the resources listed above is subject to the terms of each provider.

        Returns
        -------
        pandas.DataFrame
            Data frame conatining data about links to other data sources.
        """
        return self._links
    
    def get_movies(self) -> pd.DataFrame:
        """
        Provides table in which each represents one movie, and has the following format:

            `movieId`, `title`, `genres`

        Movie titles are entered manually or imported from https://www.themoviedb.org/,
        and include the year of release in parentheses. Errors and inconsistencies may
        exist in these titles.

        Genres are a pipe-separated list, and are selected from the following:

        * Action
        * Adventure
        * Animation
        * Children's
        * Comedy
        * Crime
        * Documentary
        * Drama
        * Fantasy
        * Film-Noir
        * Horror
        * Musical
        * Mystery
        * Romance
        * Sci-Fi
        * Thriller
        * War
        * Western
        * (no genres listed)

        Returns
        -------
        pandas.DataFrame
            Data frame conatining information about movies.
        """
        return self._movies

    def get_ratings(self) -> pd.DataFrame:
        """
        Provides table in which each row represents one rating of one movie
        by one user, and has the following format:

            `userId`, `movieId`, `rating`, `timestamp`

        The rows are ordered first by `userId`, then, within user, by `movieId`.

        Ratings are made on a 5-star scale, with half-star increments
        (0.5 stars - 5.0 stars).

        Timestamps represent seconds since midnight Coordinated Universal
        Time (UTC) of January 1, 1970.

        Returns
        -------
        pandas.DataFrame
            Data frame conatining data about user ratings.
        """
        return self._ratings
    
    def get_tags(self) -> pd.DataFrame:
        """
        Provides table in which each row represents one tag applied to
        one movie by one user, and has the following format:

            `userId`, `movieId`, `tag`, `timestamp`

        The lines within this file are ordered first by userId,
        then, within user, by movieId.

        Tags are user-generated metadata about movies. Each tag is
        typically a single word or short phrase. The meaning, value,
        and purpose of a particular tag is determined by each user.

        Timestamps represent seconds since midnight Coordinated
        Universal Time (UTC) of January 1, 1970.

        Returns
        -------
        pandas.DataFrame
            Data frame conatining information about tags.
        """
        return self._tags
    
    def get_movie_by_id(self, movie_id: int):
        """
        Provides movie data from the dataset for a given movie id.

        Parameters
        ----------
        movie_id : int
            Id of the movie that should be retrieved.

        Returns
        -------
        Movie
            Contains properties `id`, `title` and `tags`
            (in form of a list).
        
        Raises
        ------
        dataset.InvalidMovieException
            When there is no movie with requested `movieId`.
        """

        if movie_id not in self._movies.index:
            raise InvalidMovieException(f'There is no movie with movieId={movie_id}.')
        
        title: str
        tags_str: str
        title, tags_str = self._movies.loc[[movie_id]].values.reshape(-1)
        tags = tags_str.split('|')
        return Movie(movie_id, title, tags)

    def delete_rating(self, user_id: int, movie_id: int) -> bool:
        """
        Deletes a requested user rating from the `ratings` table.

        Parameters
        ----------
        user_id : int
            Id of the user whose rating should be removed.

        movie_id : int
            Id of the movie for which the rating should be removed.
        
        Returns
        -------
        bool
            `True` if the rating was actually in the table `ratings`,
            `False` otherwise.
        
        Raises
        ------
        dataset.InvalidUserException
            When there is no user with requested `userId`.
        
        dataset.InvalidMovieException
            When there is no movie with requested `movieId`.
        """

        if user_id not in self._ratings.index.get_level_values('userId'):
            raise InvalidUserException(f'There is no user with userId={user_id}.')
        
        if movie_id not in self._movies.index:
            raise InvalidMovieException(f'There is no movie with movieId={movie_id}.')
        
        if (user_id, movie_id) in self._ratings.index:
            self._ratings.drop([user_id, movie_id], inplace=True)
            return True
        
        return False