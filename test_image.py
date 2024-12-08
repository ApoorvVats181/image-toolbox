import unittest
import numpy as np
import os
from image_toolbox.image import Image
from PIL import Image as PILImage

class TestImageClass(unittest.TestCase):

    def setUp(self):
        # Create a simple 3x3 image for testing (RGB)
        self.img_data = np.zeros((3, 3, 3), dtype=np.uint8)
        self.img_data[1, 1] = [255, 0, 0]  # Red pixel at center
        test_image_path = 'test_image.jpg'
        PILImage.fromarray(self.img_data.astype(np.uint8)).save(test_image_path)
        self.test_image_path = test_image_path

    def test_load(self):
        img = Image(self.test_image_path)
        self.assertEqual(img.width, 3)  # Image width should be 3
        self.assertEqual(img.height, 3)  # Image height should be 3
        self.assertEqual(img.image_data.shape, (3, 3, 3))  # Image data should be 3x3 RGB

    def test_save(self):
        img = Image(self.test_image_path)
        img.save('saved_image.jpg')
        self.assertTrue(os.path.exists('saved_image.jpg'))  # Image should be saved
        os.remove('saved_image.jpg')  # Clean up

    def test_show(self):
        img = Image(self.test_image_path)
        # We won't actually test the plot in this case, but if `show()` runs without errors, it's fine
        try:
            img.show()
        except Exception as e:
            self.fail(f"show() raised {type(e).__name__} unexpectedly!")

    def test_repr(self):
        img = Image(self.test_image_path)
        self.assertEqual(repr(img), "Image(width=3, height=3)")  # Check string representation

    def tearDown(self):
        # Clean up the test image file
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
