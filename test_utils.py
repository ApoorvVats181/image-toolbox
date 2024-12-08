import unittest
import os
import numpy as np
from image_toolbox.utils import load_image, save_image

class TestUtils(unittest.TestCase):

    def test_load_image(self):
        # Create a simple 1x1 image for testing
        test_image_path = 'test_image.jpg'
        test_image_data = np.array([[255, 0, 0]])  # Red pixel (RGB)
        PILImage.fromarray(test_image_data.astype(np.uint8)).save(test_image_path)

        # Test loading the image
        loaded_image = load_image(test_image_path)
        self.assertEqual(loaded_image.shape, (1, 1, 3))  # Should load as a 1x1 RGB image

        # Clean up
        os.remove(test_image_path)

    def test_save_image(self):
        # Create a simple 1x1 image
        test_image_data = np.array([[255, 0, 0]])  # Red pixel (RGB)

        # Save the image
        save_image(test_image_data, 'output_image.jpg')
        
        # Check if the image was saved correctly
        self.assertTrue(os.path.exists('output_image.jpg'))
        
        # Clean up
        os.remove('output_image.jpg')
