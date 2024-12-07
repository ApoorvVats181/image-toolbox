import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

class Image:
    def __init__(self, image_path=None):
        self.image_data = None
        self.width = None
        self.height = None
        if image_path:
            self.load(image_path)

    def load(self, image_path):
        """ Load an image from a file path. """
        img = PILImage.open(image_path)
        self.image_data = np.array(img)
        self.width, self.height = img.size

    def save(self, save_path):
        """ Save the image data to a file. """
        img_data = np.clip(self.image_data, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_data)
        img.save(save_path)

    def show(self):
        """ Display the image using matplotlib. """
        plt.imshow(self.image_data)
        plt.axis('off')
        plt.show()

    def __repr__(self):
        """ Return a string representation of the Image object. """
        return f"Image(width={self.width}, height={self.height})"
