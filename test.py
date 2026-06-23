import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fusionet2 import FusionModel
from dataset import VesselDataset
import os
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image

def decode_mask(mask, colors, selected_ids=None):
    # mask: (H, W) indices
    # colors: list of (R, G, B)
    # selected_ids: list of ids to include, if None include all 
    h, w = mask.shape
    rgb = torch.zeros((3, h, w), dtype=torch.float32)
    for i, color in enumerate(colors):
        if selected_ids is not None and i not in selected_ids:
            continue
        # color is (R, G, B)
        idx = (mask == i)
        rgb[0][idx] = color[0] / 255.0
        rgb[1][idx] = color[1] / 255.0
        rgb[2][idx] = color[2] / 255.0
    return rgb

def test(model=None):
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 7  # 0:bg, 1:noise, 2-6: 5 vessel classes (noise now predicted directly)
    
    # Paths
    data_root = 'DSCA_new'
    model_path = 'best_fusion_model.pth'
    output_base_dir = 'results'

    # Subdirectories for different branches
    dirs = {
        'overall': os.path.join(output_base_dir, 'overall'),
        'main': os.path.join(output_base_dir, 'main'),
        'cerebral': os.path.join(output_base_dir, 'cerebral'),
        'noise': os.path.join(output_base_dir, 'noise')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Model
    if model is None:
        print("Initializing model and loading weights...")
        model = FusionModel(num_classes=num_classes)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Warning: {model_path} not found. Running with random weights.")
        model = model.to(device)
    
    model.eval()
    
    # Data
    print("Loading data...")
    val_dataset = VesselDataset(data_root, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Inference
    with torch.no_grad():
        for i, (image, masks_dict, img_name) in enumerate(tqdm(val_loader)):
            image = image.to(device)
            
            output, _ = model(image)
            
            # Save result
            save_name = img_name[0]
            
            # Get original size
            img_path = os.path.join(val_dataset.images_dir, save_name)
            with Image.open(img_path) as original_img:
                orig_w, orig_h = original_img.size

            # Resize logits to original size BEFORE argmax for smoother boundaries
            output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_logits = torch.softmax(output, dim=1)
            pred = torch.argmax(pred_logits, dim=1) # (1, H, W), IDs in [0..6]
            
            # 1. Overall Result (Fused) - includes all 7 classes
            pred_rgb_overall = decode_mask(pred[0].cpu(), val_dataset.colors)
            vutils.save_image(pred_rgb_overall, os.path.join(dirs['overall'], save_name))
            
            # 2. Main Result (Carotid/Vertebral: 2, 3)
            pred_rgb_main = decode_mask(pred[0].cpu(), val_dataset.colors, selected_ids=[2, 3])
            vutils.save_image(pred_rgb_main, os.path.join(dirs['main'], save_name))
            
            # 3. Cerebral Result (4, 5, 6)
            pred_rgb_cerebral = decode_mask(pred[0].cpu(), val_dataset.colors, selected_ids=[4, 5, 6])
            vutils.save_image(pred_rgb_cerebral, os.path.join(dirs['cerebral'], save_name))

            # 4. Noise Result (1 only)
            pred_rgb_noise = decode_mask(pred[0].cpu(), val_dataset.colors, selected_ids=[1])
            vutils.save_image(pred_rgb_noise, os.path.join(dirs['noise'], save_name))
            
    print(f"Results saved to {output_base_dir}")

if __name__ == '__main__':
    test()
