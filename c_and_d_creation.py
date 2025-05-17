import os
import shutil
import random

# Define base directories
base_dir_c = './C'
base_dir_d = './D'

# Create directories C and D with images and masks subfolders
os.makedirs(os.path.join(base_dir_c, 'images'), exist_ok=True)
os.makedirs(os.path.join(base_dir_c, 'masks'), exist_ok=True)
os.makedirs(os.path.join(base_dir_d, 'images'), exist_ok=True)
os.makedirs(os.path.join(base_dir_d, 'masks'), exist_ok=True)

# Source directory where images and masks are saved from previous code
source_dir = './oxford_cats'
source_images_dir = os.path.join(source_dir, 'images')
source_masks_dir = os.path.join(source_dir, 'masks')

# List all images in source images directory
all_images = os.listdir(source_images_dir)

# Separate cat and dog images based on filename prefix
cat_images = [img for img in all_images if img.startswith('cat_')]
dog_images = [img for img in all_images if img.startswith('dog_')]

# Randomly select 200 cat images for directory C
cat_images_c = random.sample(cat_images, 200)

# Remaining cat images for directory D (100 cat images)
cat_images_d = [img for img in cat_images if img not in cat_images_c]
if len(cat_images_d) > 100:
    cat_images_d = random.sample(cat_images_d, 100)

# All dog images for directory C
dog_images_c = dog_images

# Function to copy images and masks to target directory
def copy_images_and_masks(image_list, target_dir):
    for img_name in image_list:
        # Copy image
        src_img_path = os.path.join(source_images_dir, img_name)
        dst_img_path = os.path.join(target_dir, 'images', img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        # Corresponding mask name (same name but .png extension)
        mask_name = img_name.replace('.jpg', '.png')
        src_mask_path = os.path.join(source_masks_dir, mask_name)
        dst_mask_path = os.path.join(target_dir, 'masks', mask_name)
        shutil.copy(src_mask_path, dst_mask_path)

# Copy cat and dog images to directory C
copy_images_and_masks(cat_images_c, base_dir_c)
copy_images_and_masks(dog_images_c, base_dir_c)

# Copy 100 cat images to directory D
copy_images_and_masks(cat_images_d, base_dir_d)

print(f"Directory C: {len(cat_images_c)} cat images and {len(dog_images_c)} dog images")
print(f"Directory D: {len(cat_images_d)} cat images")
