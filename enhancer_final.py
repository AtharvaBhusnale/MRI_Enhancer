import os
import random
import shutil

import numpy as np
import torch

# Fix for torchvision > 0.15 removing functional_tensor which basicsr uses
try:
    from torchvision.transforms import functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    import sys
    import types
    
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer


def enhance_image(model_path, input_path, output_path, scale=4):
    """
    Enhances a low-resolution image using a pre-trained Real-ESRGAN model.
    """
    print("Setting up the Real-ESRGAN model...")

    # Determine the device to use (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Define the model architecture (x4 model)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    # Set up the upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        dni_weight=None,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )

    print(f"Loading image from: {input_path}")
    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
        print(f"  -> Original dimensions: {img.width}x{img.height}")

    except FileNotFoundError:
        print(f"Error: Input image not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Enhancing image... This may take a moment.")
    try:
        output_np, _ = upsampler.enhance(img_np, outscale=scale)
        output_img = Image.fromarray(output_np)
        print(f"  -> Enhanced dimensions: {output_img.width}x{output_img.height}")

        output_img.save(output_path)
        print(f"Successfully saved enhanced image to: {output_path}")

    except Exception as e:
        print(f"An error occurred during the enhancement process: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # --- 1. Define All Paths (Matching your folders) ---
    model_file = os.path.join("weights", "RealESRGAN_x4plus.pth")
    lr_folder = "test_images_lr"
    hr_folder = "test_images_hr"

    output_folder_base = "test_images_output"
    output_folder_hr = os.path.join(output_folder_base, "original_ground_truth")
    output_folder_enhanced = os.path.join(output_folder_base, "enhanced_result")

    os.makedirs(output_folder_hr, exist_ok=True)
    os.makedirs(output_folder_enhanced, exist_ok=True)

    # --- 2. Select a Random Low-Res Image ---
    try:
        # We are looking for the _LR_x2.png files
        scale_factor = 2

        # --- THIS IS THE FIX ---
        # The suffix must be lowercase to match the .lower() check
        suffix = f"_lr_x{scale_factor}.png"
        # --- END OF FIX ---

        all_lr_images = [f for f in os.listdir(lr_folder) if f.lower().endswith(suffix)]

        if not all_lr_images:
            raise IndexError(f"No '{suffix}' files found in '{lr_folder}'.")

        random_lr_image_name = random.choice(all_lr_images)
        input_file_lr = os.path.join(lr_folder, random_lr_image_name)

        print(f"Randomly selected image: {random_lr_image_name}")

    except (FileNotFoundError, IndexError) as e:
        print(f"Error: {e}")
        print("Please run 'prepare_data.py' first.")
        exit()

    # --- 3. Find the Matching High-Res (Ground Truth) Image ---
    try:
        # We replace the suffix to get the base name
        # We must use the *original* (case-sensitive) random name

        # --- THIS IS THE FIX ---
        # We must replace the suffix that matches the file's *actual* casing.
        # We can't just replace the lowercase suffix.
        # We'll search for the suffix case-insensitively and remove it.

        # Find the part of the filename that matches the suffix, ignoring case
        suffix_to_remove = f"_LR_x{scale_factor}.png"
        index = random_lr_image_name.lower().find(suffix.lower())
        base_name = random_lr_image_name[:index]
        # --- END OF FIX ---

        hr_filename = f"{base_name}_HR.png"
        input_file_hr = os.path.join(hr_folder, hr_filename)

        if not os.path.exists(input_file_hr):
            # Try again, just in case the original filename was different
            base_name_simple = random_lr_image_name.replace(
                f"_LR_x{scale_factor}.png", ""
            )
            hr_filename = f"{base_name_simple}_HR.png"
            input_file_hr = os.path.join(hr_folder, hr_filename)

            if not os.path.exists(input_file_hr):
                raise FileNotFoundError(
                    f"Matching HR file not found at: {input_file_hr}"
                )
            else:
                base_name = base_name_simple  # Update base_name if we found it this way

    except Exception as e:
        print(f"Error finding matching HR file: {e}")
        exit()

    # --- 4. Define Final Output Paths ---
    output_file_enhanced = os.path.join(
        output_folder_enhanced, f"{base_name}_Enhanced.png"
    )
    output_file_hr_copy = os.path.join(output_folder_hr, hr_filename)

    # --- 5. Copy the Original HR Image for Comparison ---
    try:
        shutil.copyfile(input_file_hr, output_file_hr_copy)
        print(f"Copied original HR image to: {output_file_hr_copy}")
    except Exception as e:
        print(f"Error copying original HR file: {e}")
        exit()

    # --- 6. Run the Enhancement ---
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at '{model_file}'.")
    else:
        enhance_image(
            model_path=model_file,
            input_path=input_file_lr,
            output_path=output_file_enhanced,
        )
        print(f"\nComparison files are ready in '{output_folder_base}'.")
