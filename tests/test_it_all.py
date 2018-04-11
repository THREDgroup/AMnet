import unittest
import AMnet.training


class Test(unittest.TestCase):

    def test_training(self):
        with self.subTest():
            output = AMnet.training.train_geometry_autoencoder(1, 4, True, False)
            self.assertEqual(output > 0, True)
        with self.subTest():
            output = AMnet.training.train_forward_network(1, 4, True, False)
            self.assertEqual(output > 0, True)
