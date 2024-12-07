import numpy as np
from scipy.ndimage import convolve

def rotate(image_data, angle):
    """ Rotate the image by a specified angle (in degrees). """
    from scipy.ndimage import rotate
    return rotate(image_data, angle, reshape=True)

def crop(image_data, crop_width, crop_height):
    """ Crop the central part of the image based on specified width and height. """
    img_height, img_width = image_data.shape[:2]
    
    # Calculate the center coordinates of the image
    center_x, center_y = img_width // 2, img_height // 2
    
    # Calculate the top-left corner of the crop
    top_left_x = center_x - crop_width // 2
    top_left_y = center_y - crop_height // 2
    
    # Calculate the bottom-right corner of the crop
    bottom_right_x = top_left_x + crop_width
    bottom_right_y = top_left_y + crop_height
    
    # Ensure the crop area is within the image boundaries
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)
    bottom_right_x = min(bottom_right_x, img_width)
    bottom_right_y = min(bottom_right_y, img_height)
    
    # Perform the crop
    cropped_image = image_data[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return cropped_image


def resize(image_data, new_width, new_height):
    """ Resize the image to a new width and height. """
    from skimage.transform import resize
    resized_img = resize(image_data, (new_height, new_width), anti_aliasing=True)
    return np.clip(resized_img * 255, 0, 255).astype(np.uint8)

def stretch(image_data, new_width, new_height):
    """ Stretch or compress the image to fit the given dimensions. """
    return np.array(PILImage.fromarray(image_data).resize((new_width, new_height), resample=PILImage.LANCZOS), dtype=np.uint8)

def to_grayscale(image_data):
    """ Convert the image to grayscale. """
    return np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])

def blur(image_data, kernel_size=5):
    """ Apply a blur effect using a simple averaging kernel. """
    # Create a kernel with equal weights (averaging kernel)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # For RGB images, apply the blur to each channel separately
    if len(image_data.shape) == 3:
        # Apply convolution for each color channel
        blurred_img = np.stack([convolve(image_data[:,:,i], kernel) for i in range(image_data.shape[2])], axis=-1)
    else:
        # For grayscale images, apply convolution directly
        blurred_img = convolve(image_data, kernel)
    
    return blurred_img

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

def sharpen(image_data, sigma=1, strength=1):
    """ Apply a sharpening effect using a kernel and reduce noise by smoothing first. """
    
    # Sharpening kernel (3x3 kernel is commonly used)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    
    # Apply Gaussian blur before sharpening to reduce noise
    smoothed_img = gaussian_filter(image_data, sigma=sigma)

    # For RGB images, apply the sharpen effect to each channel separately
    if len(image_data.shape) == 3:
        sharpened_img = np.stack([convolve(smoothed_img[:,:,i], kernel, mode='nearest') for i in range(image_data.shape[2])], axis=-1)
    else:
        # For grayscale images, apply convolution directly
        sharpened_img = convolve(smoothed_img, kernel, mode='nearest')
    
    # Normalize and clip the pixel values to the range [0, 255] to avoid artifacts
    sharpened_img = np.clip(sharpened_img, 0, 255)

    # Apply strength factor to adjust how much sharpening is applied
    sharpened_img = image_data + strength * (sharpened_img - image_data)
    
    # Ensure the output image has valid uint8 type and values in [0, 255]
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    
    return sharpened_img

def flip(image_data, direction='horizontal'):
    """ Flip the image horizontally or vertically. """
    if direction == 'horizontal':
        return np.fliplr(image_data)
    elif direction == 'vertical':
        return np.flipud(image_data)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

