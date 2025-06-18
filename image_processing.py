import numpy as np
import multiprocessing
from scipy import ndimage
import time # For internal timing if needed, but primarily used in test harness

def _apply_morphological_operation_single_image(args):
    """
    Applies a single morphological operation (dilation or erosion) to a single binary image.
    This function is designed to be used by multiprocessing.Pool.map.

    Args:
        args (tuple): A tuple containing:
            - image (np.ndarray): The 2D binary image (0s and 1s) to process.
            - operation_type (str): The type of morphological operation ('dilation' or 'erosion').
            - kernel_size (int): The size of the square kernel (e.g., 3 for 3x3). Must be odd.

    Returns:
        np.ndarray: The processed 2D binary image.
    """
    image, operation_type, kernel_size = args

    # Validate kernel size (must be odd)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Create a square structuring element (kernel) filled with ones.
    # This defines the shape and size of the neighborhood for the operation.
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)

    if operation_type == 'dilation':
        # Binary dilation expands the white (1) regions of the image.
        # A pixel in the output is 1 if at least one pixel under the kernel
        # in the input image is 1.
        processed_image = ndimage.binary_dilation(image, structure=kernel).astype(image.dtype)
    elif operation_type == 'erosion':
        # Binary erosion shrinks the white (1) regions of the image.
        # A pixel in the output is 1 only if all pixels under the kernel
        # in the input image are 1.
        processed_image = ndimage.binary_erosion(image, structure=kernel).astype(image.dtype)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}. Choose 'dilation' or 'erosion'.")

    return processed_image

def process_images(image_list, operation_type, kernel_size, num_processes=None):
    """
    Processes a collection of binary images using the specified morphological operation
    and kernel size, leveraging multiprocessing for parallel execution.

    Args:
        image_list (list): A list of 2D NumPy arrays, where each array is a binary image.
        operation_type (str): The type of morphological operation ('dilation' or 'erosion').
        kernel_size (int): The size of the square kernel (e.g., 3 for 3x3). Must be odd.
        num_processes (int, optional): The number of parallel processes to use.
                                       If None, uses the number of CPU cores.

    Returns:
        list: A list of processed 2D NumPy arrays.
    """
    if not image_list:
        return []

    # Prepare arguments for each image to be processed by the multiprocessing pool.
    # Each item in the `tasks` list will be passed as `args` to `_apply_morphological_operation_single_image`.
    tasks = [(image, operation_type, kernel_size) for image in image_list]

    # Initialize a multiprocessing Pool.
    # The 'with' statement ensures the pool is properly closed after use.
    # If num_processes is None, it defaults to os.cpu_count().
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.map to apply the _apply_morphological_operation_single_image function
        # to each item in the `tasks` list in parallel.
        processed_images = pool.map(_apply_morphological_operation_single_image, tasks)

    return processed_images

if __name__ == '__main__':
    # This block demonstrates a very basic inline test for the module itself
    # It's primarily for quick verification, the main test harness is in test_harness.py
    print("Running a quick internal test of image_processing.py...")

    # Create a dummy binary image
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
    print(f"Time taken for single dilation: {end_time - start_time:.4f} seconds")

    # Process it with erosion
    start_time = time.time()
    eroded_image = process_images([dummy_image], 'erosion', 3, num_processes=1)[0]
    end_time = time.time()

    print("\nEroded Image (3x3 kernel):\n", eroded_image)
    print(f"Time taken for single erosion: {end_time - start_time:.4f} seconds")

    print("\nInternal test complete.")