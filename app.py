import os
import matplotlib.pyplot as plt
from image_toolbox.image import Image
from image_toolbox.transformations import rotate, crop, resize, to_grayscale, blur, sharpen, flip
from image_toolbox.utils import save_image

# Set the image path directly (you can update this path as needed)
image_path = "input_image.jpg"  

# Check if the image file exists
if not os.path.isfile(image_path):
    print(f"Error: The file {image_path} does not exist.")
    exit(1)

# Create an Image object and load the image
img = Image(image_path)

# Show the original image
#img.show()

# Rotate the image by 45 degrees
rotated_img = rotate(img.image_data, 45)
rotated_image = Image()
rotated_image.image_data = rotated_img
save_image(rotated_image.image_data, "rotated_image.jpg")

# Crop the central part of the image (crop size: 300x300)
cropped_img = crop(img.image_data, crop_width=300, crop_height=300)
cropped_image = Image()
cropped_image.image_data = cropped_img
save_image(cropped_image.image_data, "cropped_center_image.jpg")

# Resize the image to new dimensions (256x256)
resized_img = resize(img.image_data, 256, 256)
resized_image = Image()
resized_image.image_data = resized_img
save_image(resized_image.image_data, "resized_image.jpg")

# Convert the image to grayscale
gray_img = to_grayscale(img.image_data)
gray_image = Image()
gray_image.image_data = gray_img
save_image(gray_image.image_data, "grayscale_image.jpg")

# Apply a blur effect with a kernel
blurred_img = blur(img.image_data, kernel_size=12)
blurred_image = Image()
blurred_image.image_data = blurred_img
save_image(blurred_image.image_data, "blurred_image.jpg")

# Sharpen the image
sharpened_img = sharpen(img.image_data, sigma=0.0, strength=0.2)  # Adjust sigma and strength as needed
sharpened_image = Image()
sharpened_image.image_data = sharpened_img
save_image(sharpened_image.image_data, "sharpened_image.jpg")


# Flip the image horizontally
flipped_img = flip(img.image_data, 'horizontal')
flipped_image = Image()
flipped_image.image_data = flipped_img
save_image(flipped_image.image_data, "flipped_image.jpg")

# Show one of the transformed images (e.g., resized image)
#plt.imshow(resized_img)
#plt.axis('off')  # Hide the axes
#plt.show()
