import unittest
import mock
import pandas as pd
import dm_project2.predict as predict


class TestPredict(unittest.TestCase):

    def test_round_rating(self):
        self.assertEqual(4.0, predict.round_rating(3.9))
        self.assertEqual(4.5, predict.round_rating(4.25))
        self.assertEqual(0.5, predict.round_rating(0.7499))
