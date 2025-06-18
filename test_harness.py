import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image # Still used for general image generation/saving robustness
import cv2 # For image reading, grayscale conversion, and thresholding

import image_processing # Import the module we just created

def generate_and_save_random_binary_images(directory, num_images, shape=(100, 100), density=0.2):
    """
    Generates random binary images and saves them as PNG files in the specified directory.
    These images will have pixel values of 0 or 255.

    Args:
        directory (str): The path to the directory where images will be saved.
        num_images (int): The number of images to generate.
        shape (tuple): The (height, width) of each image.
        density (float): The probability of a pixel being 1 (white).
    """
    os.makedirs(directory, exist_ok=True) # Ensure the directory exists
    print(f"Generating {num_images} random binary images and saving to '{directory}'...")
    for i in range(num_images):
        image_data = (np.random.rand(*shape) < density).astype(np.uint8) * 255 # Scale to 0-255 for PNG
        img = Image.fromarray(image_data)
        file_path = os.path.join(directory, f"binary_image_{i+1:04d}.png")
        img.save(file_path)
    print(f"Generated and saved {num_images} binary images to '{directory}'.")

def generate_and_save_random_color_images(directory, num_images, shape=(100, 100)):
    """
    Generates random color images and saves them as PNG files in the specified directory.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Generating {num_images} random color images and saving to '{directory}'...")
    for i in range(num_images):
        # Create a random RGB image
        image_data = np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
        img = Image.fromarray(image_data, 'RGB')
        file_path = os.path.join(directory, f"color_image_{i+1:04d}.png")
        img.save(file_path)
    print(f"Generated and saved {num_images} color images to '{directory}'.")


def load_images_from_folder(directory):
    """
    Loads all images from a specified directory, converting them to binary (0s and 1s)
    for processing, and also returning the raw original images for display.

    If an image is color, it's converted to grayscale then binary using Otsu's thresholding.
    If an image is already grayscale, it checks if it's already effectively binary (0/255)
    or applies Otsu's if it contains intermediate grayscale values.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        tuple: A tuple containing two lists:
               - binary_images_for_processing (list): List of 2D NumPy arrays (0s and 1s).
               - raw_images_for_display (list): List of original 2D/3D NumPy arrays (as read by cv2).
    """
    binary_images_for_processing = []
    raw_images_for_display = []
    print(f"Loading images from '{directory}'...")
    # List all files in the directory, supporting more common image extensions
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            file_path = os.path.join(directory, filename)
            try:
                # Read the image using OpenCV as is (color or grayscale)
                # IMREAD_UNCHANGED flag ensures original number of channels if possible
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                if img is None:
                    print(f"Warning: Could not load image from {file_path}. Skipping.")
                    continue

                # Store the raw image for display purposes *before* any conversion
                raw_images_for_display.append(img)

                # Now, determine how to convert this image to binary (0/1) for processing
                if len(img.shape) == 3: # Color image (BGR or BGRA)
                    print(f"  Converting color image '{filename}' to binary...")
                    # Convert to grayscale
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Convert the grayscale image to binary using Otsu's thresholding
                    _, binary_img_255 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif len(img.shape) == 2: # Already grayscale or effectively binary (0-255)
                    # Check if it's already "binary" (i.e., only 0 and 255 values, or very close)
                    # We check if the image has at most two unique values, and if those values
                    # are close to 0 and 255.
                    unique_vals = np.unique(img)
                    # A small tolerance for floating point images or slightly off 0/255 values
                    if len(unique_vals) <= 2 and (np.isclose(unique_vals, 0).any() or np.isclose(unique_vals, 255).any()):
                        print(f"  Image '{filename}' is already binary (0/255). No Otsu's thresholding applied.")
                        binary_img_255 = img
                    else:
                        print(f"  Converting grayscale image '{filename}' to binary using Otsu's...")
                        # It's grayscale with intermediate values, apply Otsu's
                        _, binary_img_255 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    print(f"Warning: Unsupported image format/dimensions for {file_path}. Skipping.")
                    # If skipping for processing, also remove from raw_images_for_display to keep lists in sync
                    raw_images_for_display.pop()
                    continue

                # Convert the 0/255 binary image to 0/1 format, which is expected by our processing functions
                final_binary_img = (binary_img_255 > 0).astype(np.uint8)
                binary_images_for_processing.append(final_binary_img)

            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                # If an error occurs, remove the last added raw image to keep lists in sync
                if raw_images_for_display:
                    raw_images_for_display.pop()
    print(f"Loaded {len(binary_images_for_processing)} images for processing and {len(raw_images_for_display)} raw images for display from '{directory}'.")
    return binary_images_for_processing, raw_images_for_display

def save_images_to_folder(image_list, directory, prefix="processed_"):
    """
    Saves a list of NumPy arrays (assumed 0s and 1s binary images) as PNG files
    in the specified directory.

    Args:
        image_list (list): A list of 2D NumPy arrays to save.
        directory (str): The path to the directory where images will be saved.
        prefix (str): A prefix for the filenames of the saved images.
    """
    os.makedirs(directory, exist_ok=True) # Ensure the directory exists
    print(f"Saving {len(image_list)} processed images to '{directory}'...")
    for i, image_data in enumerate(image_list):
        # Scale back to 0-255 for PNG saving, as our internal processing uses 0/1
        img = Image.fromarray((image_data * 255).astype(np.uint8))
        file_path = os.path.join(directory, f"{prefix}{i+1:04d}.png")
        img.save(file_path)
    print(f"Saved {len(image_list)} processed images to '{directory}'.")

def display_images(raw_original_images, processed_images, operation_type, kernel_size, num_images_to_show=2):
    """
    Displays a few original (raw, unconverted) and processed (binary 0s and 1s) images side-by-side.

    Args:
        raw_original_images (list): List of original 2D/3D NumPy arrays (as read by cv2).
        processed_images (list): List of processed 2D binary images (0s and 1s).
        operation_type (str): The morphological operation performed.
        kernel_size (int): The size of the kernel used.
        num_images_to_show (int): How many pairs of images to display.
    """
    num_to_show = min(num_images_to_show, len(raw_original_images), len(processed_images))
    if num_to_show == 0:
        print("No images to display.")
        return

    fig, axes = plt.subplots(num_to_show, 2, figsize=(10, num_to_show * 5))
    fig.suptitle(f"Image {operation_type.capitalize()} (Kernel: {kernel_size}x{kernel_size})", fontsize=16)

    # Handle case where only one image is displayed, axes might not be 2D
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_to_show):
        # Display the raw original image (can be color or grayscale)
        # Matplotlib handles BGR (OpenCV default) to RGB conversion if it's a 3-channel image
        if len(raw_original_images[i].shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(raw_original_images[i], cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(raw_original_images[i], cmap='gray') # Grayscale for 2D images
        axes[i, 0].set_title(f"Original Image {i+1}")
        axes[i, 0].axis('off')

        # Display the processed binary image (0s and 1s)
        axes[i, 1].imshow(processed_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Processed Binary Image {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """
    Main function for the test harness.
    Generates images, performs processing, times it, and displays results.
    Reads images from 'unprocessed_images' and saves to 'processed_images'.
    """
    print("--- Starting Image Processing Test Harness ---")

    # --- Configuration Parameters ---
    kernel_size = 3                  # Size of the square kernel (e.g., 3 for 3x3). Must be odd.
    operation_type = 'erosion'      # Choose 'dilation' or 'erosion'
    num_processes = os.cpu_count()   # Use all available CPU cores by default

    unprocessed_dir = "unprocessed_images"
    processed_dir = "processed_images"

    print(f"\nConfiguration:")
    print(f"  Kernel Size: {kernel_size}x{kernel_size}")
    print(f"  Operation Type: {operation_type.capitalize()}")
    print(f"  Number of Processes: {num_processes} (using os.cpu_count())")
    print(f"  Unprocessed Images Directory: '{unprocessed_dir}'")
    print(f"  Processed Images Directory: '{processed_dir}'")

    # --- 2. Load Images from the 'unprocessed_images' folder ---
    # load_images_from_folder now returns two lists: binary for processing and raw for display
    binary_images_for_processing, raw_images_for_display = load_images_from_folder(unprocessed_dir)
    if not binary_images_for_processing:
        print(f"No images found in '{unprocessed_dir}' or could not be loaded and converted. Exiting.")
        return

    # --- 3. Perform Image Processing and Measure Time ---
    print(f"\nStarting image processing ({operation_type})...")
    start_time = time.time()
    # Pass the binary images to the image_processing module
    processed_images = image_processing.process_images(
        binary_images_for_processing, operation_type, kernel_size, num_processes
    )
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Image processing completed in {processing_time:.4f} seconds for {len(binary_images_for_processing)} images.")

    # --- 4. Save Processed Images to 'processed_images' folder ---
    save_images_to_folder(processed_images, processed_dir)

    # --- 5. Display Sample Images ---
    print("\nDisplaying sample before/after images...")
    # Pass the raw original images for display
    display_images(raw_images_for_display, processed_images, operation_type, kernel_size, num_images_to_show=2)
    print("Close the image display window to continue.")

    print("\n--- Test Harness Finished ---")

if __name__ == '__main__':
    main()