from PIL import Image
from rembg import remove
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.cluster import KMeans
import os

def remove_background_and_get_mask(image_path):
    # Set a custom directory for cached files
    os.environ["XDG_CACHE_HOME"] = "models/u2net.onnx"
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    if output_image.mode == 'RGBA':
        mask = np.array(output_image)[:, :, 3] > 0
    else:
        mask = np.ones(output_image.size[::-1], dtype=bool)
    return output_image, mask


def get_product_colors(image, mask, color_count=5):
    # Convert image to numpy array
    img_array = np.array(image)
    # Get only product pixels using mask
    if img_array.shape[-1] == 4:  # RGBA
        product_pixels = img_array[mask][:, :3]
    else:  # RGB
        product_pixels = img_array[mask]
    # Remove any fully transparent or black pixels
    valid_pixels = product_pixels[np.any(product_pixels != [0, 0, 0], axis=1)]
    if len(valid_pixels) == 0:
        raise ValueError("No valid product pixels found")
    kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
    kmeans.fit(valid_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    frequencies = counts / len(labels)
    sorted_indices = np.argsort(frequencies)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_frequencies = frequencies[sorted_indices]

    return sorted_colors, sorted_frequencies


def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)


def extract_colors(images_list, color_count=2):
    color_list = []
    for image_path in images_list:
        try:
            processed_image, product_mask = remove_background_and_get_mask(image_path)
            colors, freq = get_product_colors(processed_image, product_mask, color_count)
            dominant_color = rgb_to_hex(colors[0])
            color_list.append(dominant_color)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            color_list.append(None)

    return color_list
