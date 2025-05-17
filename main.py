import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, SamModel, SamProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor
import random

print("code execution started")


TARGET_CLASS = "a photo of a cat"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAM_MODEL_NAME = "facebook/sam-vit-base"
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
EPOCHS = 30
SIMILARITY_THRESHOLD = 0.25


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ITEM_A_IMAGE = "./cat.jpg"
ITEM_A_MASK = "./cat_mask.png"
ITEM_C_DIR = "./C/"
C_PRIME_DIR = "./pseudo_labeled_c/"
ITEM_D_DIR = "./D/"
os.makedirs(C_PRIME_DIR, exist_ok=True)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

sam_model = SamModel.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
sam_processor = SamProcessor.from_pretrained(SAM_MODEL_NAME)

print("DEVICE-> ",DEVICE)

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch CPU and CUDA operations.
    
    Args:
        seed (int): The seed value to use (default: 42)
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch on CPU
    torch.manual_seed(seed)
    
    # PyTorch on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Additional settings for complete determinism
    torch.backends.cudnn.deterministic = True  # Ensures that CUDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disables CUDNN benchmarking for better reproducibility
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")



def stitch_images(item_a_image: Image.Image, item_c_image: Image.Image) -> Image.Image:
    # Resize to same height if needed
    if item_a_image.size != item_c_image.size:
        item_c_image = item_c_image.resize(item_a_image.size)

    # Stitch horizontally
    stitched = Image.new("RGB", (item_a_image.width + item_c_image.width, item_a_image.height))
    stitched.paste(item_a_image, (0, 0))
    stitched.paste(item_c_image, (item_a_image.width, 0))
    return stitched

def get_points_from_mask(mask: np.ndarray, label: int, num_points: int = 10):
    indices = np.argwhere(mask == label)
    if len(indices) == 0:
        return []
    chosen = indices[np.random.choice(len(indices), size=min(num_points, len(indices)), replace=False)]
    return [tuple(pt[::-1]) for pt in chosen]

# def prepare_points(mask_a, item_a_img, item_c_img, num_fg_bg_points=2):
#     H, W = item_a_img.size[1], item_a_img.size[0]
#     mask_np = np.array(mask_a.resize((W, H)))
#
#     # Foreground points from item A (left half)
#     fg_indices = np.argwhere(mask_np == 255)
#     bg_indices = np.argwhere(mask_np == 0)
#
#     fg_points = fg_indices[np.random.choice(len(fg_indices), num_fg_bg_points, replace=False)]
#     bg_points = bg_indices[np.random.choice(len(bg_indices), num_fg_bg_points, replace=False)]
#
#     # Offset x-coordinates of item A by 0 (left half), item C by item A's width
#     offset_x = item_a_img.size[0]
#
#     # Random points from right half (item C)
#     c_H, c_W = item_c_img.size[1], item_c_img.size[0]
#     random_fg = np.random.randint(0, c_H, size=(num_fg_bg_points, 2))
#     random_bg = np.random.randint(0, c_H, size=(num_fg_bg_points, 2))
#
#     random_fg[:, 1] = np.random.randint(0, c_W, size=num_fg_bg_points) + offset_x
#     random_bg[:, 1] = np.random.randint(0, c_W, size=num_fg_bg_points) + offset_x
#
#     # Combine all points: [y, x] format
#     all_points = np.concatenate([
#         fg_points[:, [0, 1]],
#         bg_points[:, [0, 1]],
#         random_fg[:, [0, 1]],
#         random_bg[:, [0, 1]]
#     ], axis=0)
#
#     input_points = [[ [float(p[1]), float(p[0])] for p in all_points ]]  # [[ [x1, y1], [x2, y2], ... ]]
#     input_labels = [[1]*num_fg_bg_points + [0]*num_fg_bg_points + [1]*num_fg_bg_points + [0]*num_fg_bg_points]
#
#     return input_points, input_labels


def get_clip_sim_map(img: Image.Image, text_prompt: str = "a cat", window_size: int = 64):
    H, W = img.size[1], img.size[0]
    sim_map = np.zeros((H // window_size, W // window_size))
    
    for i in range(0, H - window_size + 1, window_size):
        for j in range(0, W - window_size + 1, window_size):
            patch = img.crop((j, i, j + window_size, i + window_size))
            inputs = clip_processor(text=[text_prompt], images=patch, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = clip_model(**inputs)
            sim = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)[0].item()
            sim_map[i // window_size, j // window_size] = sim
    
    return sim_map

def prepare_points(mask_a, item_a_img, item_c_img, text_prompt = "a cat", num_fg_bg_points=5, window_size=64):
    W, H = item_a_img.size
    mask_np = np.array(mask_a.resize((W, H)))

    # Points from left half (item A)
    fg_indices = np.argwhere(mask_np == 255)
    bg_indices = np.argwhere(mask_np == 0)

    fg_points = fg_indices[np.random.choice(len(fg_indices), num_fg_bg_points, replace=False)]
    bg_points = bg_indices[np.random.choice(len(bg_indices), num_fg_bg_points, replace=False)]

    # Offset x for right half
    offset_x = W

    if item_a_img.size != item_c_img.size:
        item_c_img = item_c_img.resize(item_a_img.size)
    

    # CLIP-guided sampling on item C (right half)
    sim_map = get_clip_sim_map(item_c_img, text_prompt, window_size=window_size)

    top_fg = np.unravel_index(np.argsort(sim_map.ravel())[::-1][:num_fg_bg_points], sim_map.shape)
    top_bg = np.unravel_index(np.argsort(sim_map.ravel())[:num_fg_bg_points], sim_map.shape)

    fg_points_c = np.stack(top_fg, axis=1) * window_size + window_size // 2
    bg_points_c = np.stack(top_bg, axis=1) * window_size + window_size // 2

    # Offset C's points in x direction
    fg_points_c[:, 1] += offset_x
    bg_points_c[:, 1] += offset_x

    # Combine all [y, x] points
    all_points = np.concatenate([fg_points[:, [0,1]], bg_points[:, [0,1]], fg_points_c, bg_points_c], axis=0)
    input_points = [[ [float(x), float(y)] for y, x in all_points ]]  # Convert to [x, y]
    input_labels = [[1]*num_fg_bg_points + [0]*num_fg_bg_points + [1]*num_fg_bg_points + [0]*num_fg_bg_points]

    return input_points, input_labels


def generate_mask_for_item_c(item_a_img, mask_a, item_c_img, num_prompts=10):
    stitched = stitch_images(item_a_img, item_c_img)
    all_masks = []
    all_scores = []

    for _ in range(num_prompts):
        input_points, input_labels = prepare_points(mask_a, item_a_img, item_c_img)

        inputs = sam_processor(
            stitched,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = sam_model(**inputs)
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            scores = outputs.iou_scores.cpu()

        all_masks.extend(masks[0][0])         
        all_scores.extend(scores[0][0])       

    # Select best mask
    all_scores_tensor = torch.tensor(all_scores)
    best_idx = all_scores_tensor.argmax().item()
    best_mask_full = all_masks[best_idx].squeeze().numpy()

    # Extract right half (Item C mask)
    width = best_mask_full.shape[1]
    return best_mask_full[:, width // 2:]



def generate_all_sam_masks(image_paths, text_prompt):
    for img_path in tqdm(image_paths):

        image = Image.open(img_path).convert("RGB")
        image_shape = image.size
        image_a = Image.open(ITEM_A_IMAGE).convert("RGB")
        
        mask_a = Image.open(ITEM_A_MASK)

        mask = generate_mask_for_item_c(image_a, mask_a, image, num_prompts=10)
        save_path = os.path.join(C_PRIME_DIR, os.path.basename(img_path).replace(".jpg", "_mask.png"))
        Image.fromarray((mask * 255).astype(np.uint8)).resize(image_shape).save(save_path)
       

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir  # C/images directory
        self.mask_dir = mask_dir    # C_PRIME_DIR directory (contains masks)
        self.images = []            # Will store image filenames
        self.masks = []             # Will store corresponding mask filenames
        self.feature_extractor = feature_extractor
        
        # For CLIP similarity calculation
        transform = Compose([Resize((224,224), interpolation=Image.BICUBIC), ToTensor()])
        
        # Get text embeddings for target class
        text_inputs = clip_processor(text=[TARGET_CLASS], return_tensors="pt", padding=True).to(DEVICE)
        
        # Track selected images
        selected_count = 0
        total_count = 0
        
        print(f"Scanning mask directory: {mask_dir}")
        print(f"Looking for original images in: {image_dir}")
        
        # Process all mask files in C_PRIME_DIR
        for mask_filename in os.listdir(mask_dir):
            if mask_filename.endswith("_mask.png"):
                total_count += 1
                
                # Get corresponding original image filename
                image_filename = mask_filename.replace("_mask.png", ".jpg")
                image_path = os.path.join(image_dir, image_filename)
                
                # Skip if original image doesn't exist
                if not os.path.exists(image_path):
                    print(f"Warning: Original image {image_filename} not found for mask {mask_filename}")
                    continue
                
                try:
                    # Load original RGB image for CLIP similarity
                    image = Image.open(image_path).convert("RGB")
                    pixel_values = transform(image).unsqueeze(0).to(DEVICE)
                    
                    # Calculate CLIP similarity with target class
                    with torch.no_grad():
                        outputs = clip_model(
                            pixel_values=pixel_values, 
                            input_ids=text_inputs["input_ids"],
                            attention_mask=text_inputs["attention_mask"]
                        )
                    
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    similarity = torch.cosine_similarity(image_embeds, text_embeds).item()
                    
                    # Add to dataset if similarity exceeds threshold
                    if similarity > SIMILARITY_THRESHOLD:
                        self.images.append(image_filename)
                        self.masks.append(mask_filename)
                        selected_count += 1
                        print(f"Selected image: {image_filename}, similarity: {similarity:.4f}")
                
                except Exception as e:
                    print(f"Error processing {image_filename}: {e}")
        
        print(f"Selected {selected_count} out of {total_count} images for training.")
        if selected_count == 0:
            print("WARNING: No images selected! Check your dataset and SIMILARITY_THRESHOLD.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and mask filenames
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        
        # Load image and mask
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        
        # Process for SegFormer
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        
        # Process mask
        mask_tensor = transforms.ToTensor()(mask).squeeze().long()
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
            size=pixel_values.shape[1:], 
            mode='nearest'
        ).squeeze().long()
        
        return pixel_values, mask_tensor


def train_segformer(image_dir, mask_dir):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(SEGFORMER_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_MODEL_NAME, 
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # Create dataset
    dataset = SegDataset(image_dir, mask_dir, feature_extractor)
    
    # Check if dataset has any images
    if len(dataset) == 0:
        print("No valid images found for training. Check your dataset and SIMILARITY_THRESHOLD.")
        return
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Rest of your training code...
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, masks in tqdm(dataloader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
    
    model.save_pretrained("finetuned_segformer")


def evaluate_on_test_set(model_path, test_dir):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(SEGFORMER_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(DEVICE)
    model.eval()

    image_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")
    
    # Initialize lists to store metrics
    ious = []
    pixel_accs = []
    dice_coefs = []
    
    print("\nEvaluating model on test set...")
    for img_file in tqdm(os.listdir(image_dir)):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Load image and ground truth mask
        image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        mask_file = img_file.replace(".jpg", ".png").replace(".jpeg", ".png")
        gt_mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask not found for {img_file}")
            continue
            
        gt_mask = Image.open(gt_mask_path).convert("L")

        # Generate prediction
        encoding = feature_extractor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**encoding)
            pred_mask = torch.argmax(outputs.logits, dim=1).squeeze().cpu()

        # Resize ground truth mask to match prediction size
        gt_mask = transforms.ToTensor()(gt_mask).squeeze().long()
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(), 
            size=pred_mask.shape, 
            mode='nearest'
        ).squeeze().long()
        
        # Convert to binary masks
        pred_binary = pred_mask > 0
        gt_binary = gt_mask > 0
        
        # Calculate metrics
        # 1. IoU (Intersection over Union)
        intersection = torch.logical_and(pred_binary, gt_binary).sum().item()
        union = torch.logical_or(pred_binary, gt_binary).sum().item()
        iou = intersection / union if union > 0 else 1.0
        ious.append(iou)
        
        # 2. Pixel Accuracy
        correct_pixels = (pred_binary == gt_binary).sum().item()
        total_pixels = pred_binary.numel()
        pixel_acc = correct_pixels / total_pixels
        pixel_accs.append(pixel_acc)
        
        # 3. Dice Coefficient
        dice = (2 * intersection) / (pred_binary.sum().item() + gt_binary.sum().item()) if (pred_binary.sum().item() + gt_binary.sum().item()) > 0 else 1.0
        dice_coefs.append(dice)

    # Calculate mean metrics
    mean_iou = np.mean(ious)
    mean_pixel_acc = np.mean(pixel_accs)
    mean_dice = np.mean(dice_coefs)
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print("==============================\n")
    
    # Return metrics dictionary
    return {
        "mean_iou": mean_iou,
        "mean_pixel_acc": mean_pixel_acc,
        "mean_dice": mean_dice,
        "individual_ious": ious,
        "individual_pixel_accs": pixel_accs,
        "individual_dice_coefs": dice_coefs
    }

def visualize_segmentation_results(model_path, test_dir, num_samples=5):
    import matplotlib.pyplot as plt
    
    feature_extractor = SegformerFeatureExtractor.from_pretrained(SEGFORMER_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(DEVICE)
    model.eval()

    image_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")
    
    # Get a list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random samples
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    for img_file in image_files:
        # Load image and ground truth mask
        image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        mask_file = img_file.replace(".jpg", ".png").replace(".jpeg", ".png")
        gt_mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(gt_mask_path):
            continue
            
        gt_mask = Image.open(gt_mask_path).convert("L")
        
        # Generate prediction
        encoding = feature_extractor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**encoding)
            pred_mask = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Plot ground truth mask
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        # Plot predicted mask
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"segmentation_result_{img_file}.png")
        plt.close()
        
    print(f"Saved {len(image_files)} visualization samples.")




if __name__ == '__main__':
    set_seed(42)
    image_paths = [os.path.join(os.path.join(ITEM_C_DIR,"images"), f) for f in os.listdir(os.path.join(ITEM_C_DIR,"images")) if f.endswith(".jpg")]

    print("->Generating all SAM masks for Item C...")
    generate_all_sam_masks(image_paths, text_prompt=TARGET_CLASS)

    print("-> Fine-tuning SegFormer...")
    # Pass C/images for original images and C_PRIME_DIR for masks
    train_segformer(os.path.join(ITEM_C_DIR, "images"), C_PRIME_DIR)

    print("-> Evaluating model...")
    evaluate_on_test_set("finetuned_segformer", ITEM_D_DIR)

    print("-> Visualizing Segmentation Result")
    visualize_segmentation_results("finetuned_segformer", ITEM_D_DIR, num_samples=5)
