import unittest
import numpy as np
from image_toolbox.transformations import rotate, crop, resize, to_grayscale, blur, sharpen, flip

class TestTransformations(unittest.TestCase):

    def setUp(self):
        # Create a simple 5x5 image (RGB)
        self.img_data = np.zeros((5, 5, 3), dtype=np.uint8)
        self.img_data[2, 2] = [255, 0, 0]  # Red pixel at center

    def test_rotate(self):
        rotated_image = rotate(self.img_data, 90)
        self.assertEqual(rotated_image.shape, self.img_data.shape)  # Should maintain shape

    def test_crop(self):
        cropped_image = crop(self.img_data, 3, 3)
        self.assertEqual(cropped_image.shape, (3, 3, 3))  # Cropped to 3x3 size

    def test_resize(self):
        resized_image = resize(self.img_data, 10, 10)
        self.assertEqual(resized_image.shape, (10, 10, 3))  # Resized to 10x10

    def test_to_grayscale(self):
        grayscale_image = to_grayscale(self.img_data)
        self.assertEqual(grayscale_image.shape, (5, 5))  # Grayscale image should be 2D

    def test_blur(self):
        blurred_image = blur(self.img_data, kernel_size=3)
        self.assertEqual(blurred_image.shape, self.img_data.shape)  # Should maintain shape

    def test_sharpen(self):
        sharpened_image = sharpen(self.img_data, sigma=1, strength=1)
        self.assertEqual(sharpened_image.shape, self.img_data.shape)  # Should maintain shape

    def test_flip_horizontal(self):
        flipped_image = flip(self.img_data, 'horizontal')
        self.assertEqual(flipped_image.shape, self.img_data.shape)  # Should maintain shape
        self.assertTrue(np.array_equal(self.img_data, np.fliplr(flipped_image)))  # Check horizontal flip

    def test_flip_vertical(self):
        flipped_image = flip(self.img_data, 'vertical')
        self.assertEqual(flipped_image.shape, self.img_data.shape)  # Should maintain shape
        self.assertTrue(np.array_equal(self.img_data, np.flipud(flipped_image)))  # Check vertical flip
