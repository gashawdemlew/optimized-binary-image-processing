import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image # For reading/saving images in a more standard format

import image_processing # Import the module we just created

def generate_and_save_random_binary_images(directory, num_images, shape=(100, 100), density=0.2):
    """
    Generates random binary images and saves them as PNG files in the specified directory.

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
        file_path = os.path.join(directory, f"image_{i+1:04d}.png")
        img.save(file_path)
    print(f"Generated and saved {num_images} images to '{directory}'.")

def load_images_from_folder(directory):
    """
    Loads all PNG images from a specified directory into a list of NumPy arrays.
    Handles both grayscale and color images by converting them to binary (0s and 1s).
    Color images are first converted to grayscale, then thresholded to binary.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        list: A list of 2D NumPy arrays representing the binary images (0s and 1s).
    """
    image_list = []
    print(f"Loading images from '{directory}'...")
    # List all files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"): # Only process PNG files
            file_path = os.path.join(directory, filename)
            try:
                # Open the image. .convert('L') converts to 8-bit grayscale,
                # which handles both existing grayscale and color images.
                img = Image.open(file_path).convert('L')
                # Convert to binary (0 or 1) by thresholding.
                # Pixels with intensity > 128 become 1, others become 0.
                image_array = (np.array(img) > 128).astype(np.uint8)
                image_list.append(image_array)
            except Exception as e:
                print(f"Could not load image {filename}: {e}")
    print(f"Loaded {len(image_list)} images from '{directory}'.")
    return image_list

def save_images_to_folder(image_list, directory, prefix="processed_"):
    """
    Saves a list of NumPy arrays as PNG images in the specified directory.

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

def display_images(original_images, processed_images, operation_type, kernel_size, num_images_to_show=2):
    """
    Displays a few original and processed images side-by-side using Matplotlib.

    Args:
        original_images (list): List of original 2D binary images.
        processed_images (list): List of processed 2D binary images.
        operation_type (str): The morphological operation performed.
        kernel_size (int): The size of the kernel used.
        num_images_to_show (int): How many pairs of images to display.
    """
    num_to_show = min(num_images_to_show, len(original_images))
    if num_to_show == 0:
        print("No images to display.")
        return

    fig, axes = plt.subplots(num_to_show, 2, figsize=(10, num_to_show * 5))
    fig.suptitle(f"Image {operation_type.capitalize()} (Kernel: {kernel_size}x{kernel_size})", fontsize=16)

    # Handle case where only one image is displayed, axes might not be 2D
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_to_show):
        axes[i, 0].imshow(original_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original Image {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(processed_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Processed Image {i+1}")
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
    image_shape = (200, 200)       # Dimensions of each binary image
    image_density = 0.15           # Density of '1' pixels in generated images
    num_images_to_generate = 50    # Number of images to generate and save for testing
    kernel_size = 5                # Size of the square kernel (e.g., 3 for 3x3). Must be odd.
    operation_type = 'dilation'    # Choose 'dilation' or 'erosion'
    num_processes = os.cpu_count() # Use all available CPU cores by default

    unprocessed_dir = "unprocessed_images"
    processed_dir = "processed_images"

    print(f"\nConfiguration:")
    print(f"  Image Shape: {image_shape}")
    print(f"  Image Density: {image_density}")
    print(f"  Number of Images to Generate: {num_images_to_generate}")
    print(f"  Kernel Size: {kernel_size}x{kernel_size}")
    print(f"  Operation Type: {operation_type.capitalize()}")
    print(f"  Number of Processes: {num_processes} (using os.cpu_count())")
    print(f"  Unprocessed Images Directory: '{unprocessed_dir}'")
    print(f"  Processed Images Directory: '{processed_dir}'")


    # --- 1. Generate and Save Binary Images (for testing purposes) ---
    # This step ensures there are images in the unprocessed_images folder.
    # In a real scenario, these images would already exist.
    generate_and_save_random_binary_images(unprocessed_dir, num_images_to_generate, image_shape, image_density)

    # --- 2. Load Images from the 'unprocessed_images' folder ---
    original_images = load_images_from_folder(unprocessed_dir)
    if not original_images:
        print(f"No images found in '{unprocessed_dir}'. Exiting.")
        return

    # --- 3. Perform Image Processing and Measure Time ---
    print(f"\nStarting image processing ({operation_type})...")
    start_time = time.time()
    processed_images = image_processing.process_images(
        original_images, operation_type, kernel_size, num_processes
    )
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Image processing completed in {processing_time:.4f} seconds for {len(original_images)} images.")

    # --- 4. Save Processed Images to 'processed_images' folder ---
    save_images_to_folder(processed_images, processed_dir)

    # --- 5. Display Sample Images ---
    print("\nDisplaying sample before/after images...")
    display_images(original_images, processed_images, operation_type, kernel_size, num_images_to_show=2)
    print("Close the image display window to continue.")

    print("\n--- Test Harness Finished ---")

if __name__ == '__main__':
    main()