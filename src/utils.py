import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, dataset_path):
        self.images = []
        self.correspondances = None

        self.dataset_path = dataset_path
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path):
        for filename in sorted(os.listdir(dataset_path)):
            if filename.lower().endswith(("jpg")):
                img_path = os.path.join(dataset_path, filename)
                self.images.append(load_image(img_path))
            elif filename.lower().endswith((".pkl")):
                pickle_path = os.path.join(dataset_path, filename)
                with open(pickle_path, "rb") as pickle_file:
                    self.correspondances = pickle.load(pickle_file)

def load_image(img_path):
    """ Loads and image in RGB. """
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def compute_segmentation_metrics(predicted_mask, ground_truth_mask):
    """Compare masks of the same size; nonzero pixels are foreground."""
    predicted = np.asarray(predicted_mask)
    truth = np.asarray(ground_truth_mask)
    if predicted.ndim != 2 or truth.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("Masks must be 2D arrays with the same shape")
    predicted = predicted != 0
    truth = truth != 0
    tp = int(np.count_nonzero(np.logical_and(predicted, truth)))
    fp = int(np.count_nonzero(predicted)) - tp
    fn = int(np.count_nonzero(truth)) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "F1": f1}

def compute_gray_histogram(img_gray):
    assert len(img_gray.shape) == 2, "Not a gray image"
    
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    return cv2.normalize(hist, None, alpha=1, norm_type=cv2.NORM_L1).flatten()

def compute_color_hsv_histogram(img_hsv, bins_h=30, bins_s=32, bins_v=32):
    assert len(img_hsv.shape) == 3, "Does not seem like a HSV image"
    
    # 1D histogram for each channel respecting their original ranges
    hist_h = cv2.calcHist([img_hsv], [0], None, [bins_h], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [bins_s], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [bins_v], [0, 256])
    
    # Concatenate 3 histograms into a single 1D vector
    hist_concat = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    return cv2.normalize(hist_concat, None, alpha=1, norm_type=cv2.NORM_L1).flatten()

def compute_color_rgb_histogram(img_rgb, bins_r=32, bins_g=32, bins_b=32):
    assert len(img_rgb.shape) == 3, "Does not seem like a RGB image"

    # 1D histogram for each channel, all in range [0, 256)
    hist_r = cv2.calcHist([img_rgb], [0], None, [bins_r], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins_g], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins_b], [0, 256])

    # Concatenate 3 histograms into a single 1D vector
    hist_concat = np.concatenate([hist_r, hist_g, hist_b]).flatten()

    return cv2.normalize(hist_concat, None, alpha=1, norm_type=cv2.NORM_L1).flatten()

def _l1_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).flatten()
    return vector / (vector.sum() + 1e-12)

def _split_in_blocks(img, grid_y, grid_x):
    """Yield the grid_y x grid_x blocks of an image (row by row)."""
    height, width = img.shape[:2]
    for y in range(grid_y):
        for x in range(grid_x):
            yield img[y * height // grid_y:(y + 1) * height // grid_y,
                      x * width // grid_x:(x + 1) * width // grid_x]

def compute_hsv_grid_histogram(img_hsv, grid=4, bins_h=16, bins_s=8, bins_v=8):
    """One HSV histogram per block, all blocks concatenated."""
    block_descriptors = []
    for block in _split_in_blocks(img_hsv, grid, grid):
        hist_h = cv2.calcHist([block], [0], None, [bins_h], [0, 180])
        hist_s = cv2.calcHist([block], [1], None, [bins_s], [0, 256])
        hist_v = cv2.calcHist([block], [2], None, [bins_v], [0, 256])
        block_descriptors.append(_l1_normalize(np.concatenate([hist_h, hist_s, hist_v])))
    return _l1_normalize(np.concatenate(block_descriptors))

def visualize_histograms(dataset, idx):
    if not 0 <= idx < len(dataset.images):
        raise IndexError(f"Image index {idx} out of range for {len(dataset.images)} images")

    img_rgb = dataset.images[idx]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"Image {idx}")
    axes[0, 0].axis("off")

    gray_hist = compute_gray_histogram(img_gray)
    axes[0, 1].plot(np.arange(256), gray_hist, color="black")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].set_xlim(0, 255)

    channels = [
        ("Hue", 0, 30, 180, "red"),
        ("Saturation", 1, 32, 256, "green"),
        ("Value", 2, 32, 256, "blue"),
    ]
    channel_axes = [axes[0, 2], axes[1, 0], axes[1, 1]]
    for axis, (name, channel, bins, upper_bound, color) in zip(channel_axes, channels):
        hist = cv2.calcHist([img_hsv], [channel], None, [bins], [0, upper_bound])
        hist = cv2.normalize(hist, None, alpha=1, norm_type=cv2.NORM_L1).flatten()
        bin_centers = np.linspace(0, upper_bound, bins, endpoint=False) + upper_bound / (2 * bins)
        axis.plot(bin_centers, hist, color=color)
        axis.set_title(name)
        axis.set_xlim(0, upper_bound)

    # Third method: RGB histogram (overlaid R, G, B) in the remaining subplot
    bins_rgb = 32
    for channel, color in zip(range(3), ("red", "green", "blue")):
        hist = cv2.calcHist([img_rgb], [channel], None, [bins_rgb], [0, 256])
        hist = cv2.normalize(hist, None, alpha=1, norm_type=cv2.NORM_L1).flatten()
        bin_centers = np.linspace(0, 256, bins_rgb, endpoint=False) + 256 / (2 * bins_rgb)
        axes[1, 2].plot(bin_centers, hist, color=color)
    axes[1, 2].set_title("RGB")
    axes[1, 2].set_xlim(0, 256)
    fig.suptitle(f"Histograms for image index {idx}")
    fig.tight_layout()
    plt.show()

def visualize_histograms_rgb(dataset, idx, bins_r=32, bins_g=32, bins_b=32):
    if not 0 <= idx < len(dataset.images):
        raise IndexError(f"Image index {idx} out of range for {len(dataset.images)} images")

    img_rgb = dataset.images[idx]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Image {idx}")
    axes[0].axis("off")

    rgb_hist = compute_color_rgb_histogram(img_rgb, bins_r, bins_g, bins_b)
    # Split back the concatenated descriptor for plotting
    hist_r = rgb_hist[:bins_r]
    hist_g = rgb_hist[bins_r:bins_r + bins_g]
    hist_b = rgb_hist[bins_r + bins_g:]

    for axis, hist, bins, name, color in zip(
        axes[1:],
        (hist_r, hist_g, hist_b),
        (bins_r, bins_g, bins_b),
        ("Red", "Green", "Blue"),
        ("red", "green", "blue"),
    ):
        bin_centers = np.linspace(0, 256, bins, endpoint=False) + 256 / (2 * bins)
        axis.plot(bin_centers, hist, color=color)
        axis.set_title(name)
        axis.set_xlim(0, 256)

    fig.suptitle(f"RGB histograms for image index {idx}")
    fig.tight_layout()
    plt.show()
