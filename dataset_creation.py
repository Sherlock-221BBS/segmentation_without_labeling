import os
import numpy as np
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random






# Output directory
SAVE_DIR = "./oxford_cats"
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "masks"), exist_ok=True)

# Define constants
MAX_CAT_IMAGES = 300
MAX_DOG_IMAGES = 100

# Cat breeds as provided by you
CAT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", 
    "British_Shorthair", "Egyptian_Mau", "Maine_Coon", 
    "Persian", "Ragdoll", "Russian_Blue", "Sphynx"
]

# Load datasets separately for image labels and segmentation masks
img_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=transforms.ToTensor()
)

mask_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=False,
    transform=None  # Keep as PIL image for easier processing
)

# Initialize counters
cat_count = 0
dog_count = 0

# Helper function to check if image filename contains any cat breed name
def is_cat_image(filename):
    for breed in CAT_BREEDS:
        if breed.lower() in filename.lower():
            return True
    return False

# Process all images
for i in tqdm(range(len(img_dataset))):
    image, label = img_dataset[i]
    
    # Get the filename of the image
    filename = os.path.basename(img_dataset._images[i])  # e.g. 'Abyssinian_1.jpg'
    
    # Get the corresponding mask
    _, mask = mask_dataset[i]
    mask_np = np.array(mask)
    
    # Create binary mask (foreground=1, background=0)
    # In the Oxford-IIIT Pet dataset:
    # 1 = pet, 2 = background, 3 = boundary/outline
    binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
    binary_mask[mask_np == 1] = 255  # Set pet pixels to 255 (white)
    
    # Convert tensor to PIL image for saving
    image_pil = transforms.ToPILImage()(image)
    
    # Determine if it's a cat or dog based on filename
    if is_cat_image(filename):
        if cat_count < MAX_CAT_IMAGES:
            # Save cat image
            image_pil.save(os.path.join(SAVE_DIR, "images", f"cat_{cat_count:03d}.jpg"))
            
            # Save binary mask
            mask_pil = Image.fromarray(binary_mask)
            mask_pil.save(os.path.join(SAVE_DIR, "masks", f"cat_{cat_count:03d}.png"))
            
            cat_count += 1
    else:  # It's a dog
        if dog_count < MAX_DOG_IMAGES:
            # Save dog image
            image_pil.save(os.path.join(SAVE_DIR, "images", f"dog_{dog_count:03d}.jpg"))
            
            # Save binary mask
            mask_pil = Image.fromarray(binary_mask)
            mask_pil.save(os.path.join(SAVE_DIR, "masks", f"dog_{dog_count:03d}.png"))
            
            dog_count += 1
    
    # Check if we have collected enough images
    if cat_count >= MAX_CAT_IMAGES and dog_count >= MAX_DOG_IMAGES:
        break

print(f"Saved {cat_count} cat images and {dog_count} dog images with binary masks in '{SAVE_DIR}'")
