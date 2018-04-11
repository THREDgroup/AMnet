import unittest
import WAnet.training
import WAnet.preprocessing


class Test(unittest.TestCase):

    def test_training(self):
        with self.subTest():
            output = WAnet.preprocessing.extract_data(N=10)
            self.assertEqual(output, True)
        with self.subTest():
            output = WAnet.training.train_geometry_autoencoder(1, 4, True, False)
            self.assertEqual(output > 0, True)
        with self.subTest():
            output = WAnet.training.train_response_autoencoder(1, 4, True, False)
            self.assertEqual(output > 0, True)
        with self.subTest():
            output = WAnet.training.train_forward_network(1, 4, True, False)
            self.assertEqual(output > 0, True)
        with self.subTest():
            output = WAnet.training.train_inverse_network(1, 4, True, False)
            self.assertEqual(output > 0, True)
