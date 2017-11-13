import unittest
from src.unidentified_customer import TrainClassifier


class UnidentifiedCustomerTest(unittest.TestCase):
    def setUpClass(self, cls):
        self.unidentified_class = TrainClassifier()
