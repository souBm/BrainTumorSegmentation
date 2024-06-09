import cv2 as cv
import os
import numpy as np
from PIL import Image

# Function to apply mask on the image
def apply_mask(image, mask):
    result_img = cv.bitwise_and(image, mask)
    return result_img

# Main function
if __name__ == "__main__":
    height = 240
    width = 240

    images = []
    masks = []

    # Load test images
    image_folder = 'DATA/test-images'
    for each_image in os.listdir(image_folder):
        if each_image.endswith('.png'):
            image_path = os.path.join(image_folder, each_image)
            image = cv.imread(image_path)
            if image is not None:
                image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB)).resize((height, width))
                images.append(np.array(image))
            else:
                print(f"Failed to load image: {image_path}")

    # Load mask images 
    mask_folder = 'DATA/output_masks'
    for each_mask in os.listdir(mask_folder):
        if each_mask.endswith('.png'):
            mask_path = os.path.join(mask_folder, each_mask)
            mask = cv.imread(mask_path)
            if mask is not None:
                mask = Image.fromarray(cv.cvtColor(mask, cv.COLOR_BGR2RGB)).resize((height, width))
                mask = np.array(mask)
                masks.append(mask)
            else:
                print(f"Failed to load mask: {mask_path}")

    # Apply mask on each image
    output = []
    for img, mask in zip(images, masks):
        result = apply_mask(img, mask)
        output.append(result)

    # Save the resulting images with the applied mask
    output_folder = 'DATA/Intersected_img'
    name = "intersect_"
    idx = 1
    for output_img in output:
        cv.imwrite(os.path.join(output_folder, f'{name}{idx}.png'), output_img)
        idx += 1
    
