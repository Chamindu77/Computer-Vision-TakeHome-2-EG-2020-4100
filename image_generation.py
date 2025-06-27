import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

### -------------------- Task 1: Otsu Thresholding with Gaussian Noise --------------------

def create_synthetic_image():
    """Creates a synthetic grayscale image with 2 objects and a background."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:50, 20:50] = 100   # Object 1
    img[60:90, 60:90] = 200   # Object 2
    return img

def add_gaussian_noise(image, mean=0, std=15):
    """Adds Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_otsu_threshold(image):
    """Applies Otsu's thresholding."""
    _, otsu_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_result

def task1():
    print("[Task 1] Running Otsu’s thresholding on synthetic noisy image...")
    original = create_synthetic_image()
    noisy = add_gaussian_noise(original)
    otsu = apply_otsu_threshold(noisy)

    cv2.imwrite("output/task1_original.png", original)
    cv2.imwrite("output/task1_noisy.png", noisy)
    cv2.imwrite("output/task1_otsu_result.png", otsu)

### -------------------- Task 2: Region Growing --------------------

def get_neighbors(x, y, shape):
    """Returns 4-neighbor coordinates (within image bounds)."""
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            neighbors.append((nx, ny))
    return neighbors

def region_growing(image, seed_points, threshold=10):
    """Implements region growing for grayscale images."""
    visited = np.zeros_like(image, dtype=bool)
    output = np.zeros_like(image, dtype=np.uint8)

    for seed in seed_points:
        x, y = seed
        seed_val = image[x, y]
        queue = [(x, y)]

        while queue:
            px, py = queue.pop()
            if visited[px, py]:
                continue
            visited[px, py] = True
            diff = abs(int(image[px, py]) - int(seed_val))
            if diff <= threshold:
                output[px, py] = 255
                queue.extend(get_neighbors(px, py, image.shape))

    return output

def task2():
    print("[Task 2] Running region growing...")
    image = create_synthetic_image()
    noisy_image = add_gaussian_noise(image)

    seed_points = [(25, 25), (70, 70)]  # Inside object 1 and 2
    segmented = region_growing(noisy_image, seed_points, threshold=20)

    cv2.imwrite("output/task2_input.png", noisy_image)
    cv2.imwrite("output/task2_segmented.png", segmented)

### -------------------- Run All --------------------

if __name__ == "__main__":
    task1()
    task2()
    print("All outputs saved to the 'output' folder.")
