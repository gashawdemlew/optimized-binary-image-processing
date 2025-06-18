# Optimized Morphological Image Processing

## Project Overview

This project provides a highly optimized Python solution for performing fundamental morphological operations (specifically **binary erosion** and **binary dilation**) on collections of images. Designed with performance in mind, it efficiently processes numerous images by leveraging parallel computing and optimized numerical libraries, now primarily utilizing **OpenCV** for its core image processing functions.

## Features

* **Core Operations**: Implements both binary dilation and erosion using `cv2` (OpenCV).
* **Flexible Kernel**: Supports square kernels of any odd size (e.g., 3x3, 5x5, 7x7).
* **Automated Image Handling**:
    * Reads input images from an `unprocessed_images` directory.
    * Saves all processed images to a `processed_images` directory.
    * Automatically converts input images (color or grayscale) into a standardized binary format (0s and 1s) using a threshold, ensuring compatibility with morphological operations.
* **Performance Optimization**: Achieves significant speedups through:
    * **Multiprocessing**: Distributes image processing tasks across multiple CPU cores for parallel execution.
    * **Efficient Array Operations**: Utilizes highly optimized `NumPy` and `OpenCV` (which itself heavily relies on optimized C++ code) for rapid array computations. OpenCV's functions leverage underlying efficient algorithms and **SIMD instructions** for speed.

## How Optimization Works

The solution employs a powerful two-pronged approach to maximize performance:

1.  ### Multiprocessing (Parallel Throughput)

    The `test_harness.py` script manages a *collection* of images, and the core processing logic in `image_processing.py` utilizes Python's `multiprocessing.Pool`. This allows:
    * **Task Distribution**: Each image in the collection is treated as an independent processing task.
    * **Parallel Execution**: The `multiprocessing.Pool.map` function efficiently distributes these tasks across all available CPU cores. For instance, on a machine with 8 CPU cores, up to 8 images can be processed concurrently.
    * **Bypassing the GIL**: This approach is particularly effective for CPU-bound tasks like image processing, as each process runs in its own Python interpreter, effectively bypassing Python's Global Interpreter Lock (GIL) and enabling true parallel execution. This dramatically reduces the total time for large image batches.

2.  ### Efficient Array Operations (Per-Image Speed)

    At the heart of the morphological transformations (`cv2.dilate` and `cv2.erode`) are OpenCV's highly optimized functions. These functions, built to operate on `NumPy` arrays, are crucial because:
    * **Compiled C++ Code**: OpenCV is primarily written in optimized C++. This means the actual pixel-level operations are executed as highly efficient, compiled native code, which is significantly faster than equivalent operations written purely in Python.
    * **SIMD Instructions**: OpenCV's routines are designed to exploit **SIMD (Single Instruction, Multiple Data) instructions** available on modern CPUs. SIMD allows a single CPU instruction to perform the same operation on multiple data elements (e.g., pixels) simultaneously. This is a fundamental acceleration for array-based computations.

In essence, `multiprocessing` accelerates the *overall processing of multiple images* by running tasks in parallel, while `NumPy` and `OpenCV` ensure that *each individual image transformation* is executed with maximum speed by leveraging highly optimized, compiled code that fully utilizes modern CPU capabilities, including SIMD.

## Getting Started

### Prerequisites

* Python 3.7 or higher
* `pip` (Python package installer)
* Docker (Optional, but recommended for consistent environments)

### Project Structure

.├── image_processing.py├── test_harness.py├── requirements.txt├── Dockerfile└── README.md
### Installation

#### 1. Manual Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/optimized-image-processing.git](https://github.com/your-username/optimized-image-processing.git)
    cd optimized-image-processing
    ```
    *(Remember to replace `your-username/optimized-image-processing.git` with the actual URL of your GitHub repository.)*

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

#### 2. Docker (Recommended for a consistent and isolated environment)

1.  **Build the Docker Image**:
    ```bash
    docker build -t image-processor .
    ```

2.  **Run the Docker Container**:
    ```bash
    docker run -it image-processor
    ```
    This command will build the Docker image (if not already built) and then execute the `test_harness.py` script inside the container.

## Usage

### Running the Test Harness

The `test_harness.py` script serves as a comprehensive demonstration and validation tool for the image processing pipeline:

1.  It automatically creates an `unprocessed_images` directory (if it doesn't already exist).
2.  It generates a configurable number of random binary images into the `unprocessed_images` folder. (You can also place your own PNG images, color or grayscale, in this folder).
3.  It loads all images from `unprocessed_images`, automatically converting them to a binary (0s and 1s) format suitable for morphological operations.
4.  It applies the chosen morphological operation (dilation or erosion) using the specified kernel size across the entire collection of images.
5.  It saves all processed images into a newly created `processed_images` directory.
6.  Finally, it displays a few sample "before-and-after" image pairs for visual comparison and reports the total processing time.

To initiate the test harness:

```bash
python test_harness.py
ConfigurationYou can easily customize the behavior of the image processing pipeline by modifying parameters within the main() function of test_harness.py:image_shape: Defines the dimensions (height, width) of the images.image_density: Controls the density of '1' pixels in randomly generated images.num_images_to_generate: Specifies how many random images the script will create in unprocessed_images.kernel_size: Sets the size of the square kernel (e.g., 3 for a 3x3 kernel). Must be an odd number.operation_type: Choose between 'dilation' or 'erosion'.num_processes: Determines the number of parallel processes to utilize. Set to None to automatically use all available CPU cores.unprocessed_dir: The name of the input directory for raw images.processed_dir: The name of the output directory for processed images.Example OutputUpon successful execution of python test_harness.py, you will observe detailed progress and timing information in your terminal. Simultaneously, a Matplotlib window will appear, showcasing selected original and processed image pairs. The results will also be saved as PNG files within the processed_images folder.ContributingContributions are welcome! Feel free to open issues to report bugs or suggest features, or submit pull requests