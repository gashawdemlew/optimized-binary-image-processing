import numpy as np
import multiprocessing
import cv2 # New import for OpenCV
import time # For internal timing if needed, but primarily used in test harness

def _apply_morphological_operation_single_image(args):
    """
    Applies a single morphological operation (dilation or erosion) to a single binary image
    using OpenCV. This function is designed to be used by multiprocessing.Pool.map.

    Args:
        args (tuple): A tuple containing:
            - image (np.ndarray): The 2D binary image (values 0 and 1) to process.
            - operation_type (str): The type of morphological operation ('dilation' or 'erosion').
            - kernel_size (int): The size of the square kernel (e.g., 3 for 3x3). Must be odd.

    Returns:
        np.ndarray: The processed 2D binary image (values 0 and 1).
    """
    image, operation_type, kernel_size = args

    # Validate kernel size (must be odd)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # OpenCV's morphological operations (erode, dilate) typically expect
    # binary images with pixel values of 0 and 255.
    # Our input 'image' is in 0 and 1 format, so convert it to 0/255.
    image_for_cv2 = (image * 255).astype(np.uint8)

    # Define a square structuring element (kernel) filled with ones.
    # This defines the shape and size of the neighborhood for the operation.
    # OpenCV's `np.ones` for kernel also expects uint8.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the morphological operation
    if operation_type == 'dilation':
        # Binary dilation expands the white (255) regions of the image.
        # It helps to fill small holes and connect disconnected components.
        processed_image_cv2 = cv2.dilate(image_for_cv2, kernel, iterations=1)
    elif operation_type == 'erosion':
        # Binary erosion shrinks the white (255) regions of the image.
        # It removes small white noise and disconnects connected components.
        processed_image_cv2 = cv2.erode(image_for_cv2, kernel, iterations=1)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}. Choose 'dilation' or 'erosion'.")

    # Convert the processed image back to 0 and 1 format for consistency
    # with the rest of the project's binary image representation.
    processed_image = (processed_image_cv2 > 0).astype(np.uint8)

    return processed_image

def process_images(image_list, operation_type, kernel_size, num_processes=None):
    """
    Processes a collection of binary images using the specified morphological operation
    and kernel size, leveraging multiprocessing for parallel execution.

    Args:
        image_list (list): A list of 2D NumPy arrays, where each array is a binary image (0s and 1s).
        operation_type (str): The type of morphological operation ('dilation' or 'erosion').
        kernel_size (int): The size of the square kernel (e.g., 3 for 3x3). Must be odd.
        num_processes (int, optional): The number of parallel processes to use.
                                       If None, uses the number of CPU cores.

    Returns:
        list: A list of processed 2D NumPy arrays (0s and 1s).
    """
    if not image_list:
        return []

    # Prepare arguments for each image to be processed by the multiprocessing pool.
    tasks = [(image, operation_type, kernel_size) for image in image_list]

    # Initialize a multiprocessing Pool.
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_images = pool.map(_apply_morphological_operation_single_image, tasks)

    return processed_images

if __name__ == '__main__':
    # This block demonstrates a very basic inline test for the module itself
    print("Running a quick internal test of image_processing.py with OpenCV...")

    # Create a dummy binary image (0s and 1s)
    dummy_image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    # Process it with dilation
    start_time = time.time()
    dilated_image = process_images([dummy_image], 'dilation', 3, num_processes=1)[0]
    end_time = time.time()

    print("\nOriginal Image:\n", dummy_image)
    print("\nDilated Image (3x3 kernel):\n", dilated_image)
    print(f"Time taken for single dilation (OpenCV): {end_time - start_time:.4f} seconds")

    # Process it with erosion
    start_time = time.time()
    eroded_image = process_images([dummy_image], 'erosion', 3, num_processes=1)[0]
    end_time = time.time()

    print("\nEroded Image (3x3 kernel):\n", eroded_image)
    print(f"Time taken for single erosion (OpenCV): {end_time - start_time:.4f} seconds")

    print("\nInternal test complete.")