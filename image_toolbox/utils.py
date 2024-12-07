
import numpy as np
from PIL import Image as PILImage

def load_image(image_path):
    """ Load an image from the specified path. """
    try:
        img = PILImage.open(image_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading image: {e}")
    return np.array(img)


def save_image(image_data, output_path):
    """ Save the image data to a file. """
    # Ensure the image data is in the uint8 format (8-bit per channel)
    image_data = np.clip(image_data, 0, 255).astype(np.uint8)
    
    # Check if the image is grayscale (2D) or RGB (3D)
    if len(image_data.shape) == 2:  # Grayscale
        img = PILImage.fromarray(image_data)
    else:  # RGB or RGBA
        img = PILImage.fromarray(image_data)
    
    # Save the image
    img.save(output_path)
