# DM-project2

[![Python package](https://github.com/MichalRedm/DM-project2/actions/workflows/python-package.yml/badge.svg)](https://github.com/MichalRedm/DM-project2/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the second project for Data Mining classes at the Poznań University of Technology. The code in this repository has been created by the team *Kung Fu Pandas*, consisting of:
- [Piotr Balewski](https://github.com/PBalewski),
- [Adam Dobosz](https://github.com/addobosz),
- [Wiktor Kamzela](https://github.com/Wector1),
- [Michał Redmer](https://github.com/MichalRedm).

The general purpose of the code in this repository is to predict ratings given by different users for movies in the [Movie Lens dataset](https://grouplens.org/datasets/movielens/).

## Prequisities

To run the code from this repository, you need to have the following installed on your computer:
- [Python](https://www.python.org/downloads/) (version 3.10 or higher).

Additionally, to run code samples in the file `samples.ipynb` [Jupyer Notebook](https://jupyter.org/install) is needed (the file can alternatively be opened in [Google Colaboratory](https://colab.research.google.com/)).

## Setup

To download the respository to your local computer run the following command:

```
$ git clone https://github.com/MichalRedm/DM-project2.git
```

Then, you need to install all the Python dependencies:

```
$ pip install -r requirements.txt
```

Once this is done, you are ready to run the code.

## Preprocessing the data

The file `dm_project2/predict.py` is responsible for predicting a rating for a given user-movie pair and a given dataset:
```
$ python dm_project2/predict.py <user_id> <movie_id> <dataset_name>
```
Name of the dataset should be either `ml-latest-small` or `ml-latest`.

Example:
```
$ python dm_project2/predict.py 1 1 ml-latest-small
Predicted rating for movie "Toy Story (1995)" and user with userId=1: 4.432912849776076 (rounded: 4.5)
```

## Testing the results

To test how well our method works, use the script `dm_project2/test.py`.
```
python dm_project2/test.py <dataset_name> <number_of_samples>
```
Name of the dataset should be either `ml-latest-small` or `ml-latest`; however, testing on the larger dataset is not recommended, as it is very slow and requires high computational power.

Example:
```
$ python dm_project2/test.py ml-latest-small 10000
Model MSE:     0.8084 (standarized:  0.7450)
Baseline MSE:  0.9664 (standarized:  0.8906)
```

## Code samples

To have some insight into how our function for data preprocessing operates, visit the file [samples.ipynb](https://github.com/MichalRedm/DM-project2/blob/main/dm_project2/samples.ipynb).
