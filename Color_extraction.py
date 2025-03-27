from PIL import Image
from rembg import remove
import numpy as np
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor


def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error downloading image: {e}")


def load_image(image_path_or_url):
    return download_image(image_path_or_url) if image_path_or_url.startswith("http") else Image.open(image_path_or_url).convert("RGBA")


def process_image(image):
    output_image = remove(image)
    mask = np.array(output_image)[:, :, 3] > 0 if output_image.mode == 'RGBA' else np.ones(output_image.size[::-1], dtype=bool)
    return output_image, mask


def extract_dominant_colors(image, mask, color_count=2):
    img_array = np.array(image)
    product_pixels = img_array[mask][:, :3] if img_array.shape[-1] == 4 else img_array[mask]

    if len(product_pixels) == 0:
        return None  # No valid pixels found

    kmeans = KMeans(n_clusters=color_count, random_state=42, n_init="auto")  # Auto-tuned for efficiency
    kmeans.fit(product_pixels)
    return ['#{:02x}{:02x}{:02x}'.format(*map(int, color)) for color in kmeans.cluster_centers_]


def process_single_image(image_path_or_url, color_count):
    try:
        image = load_image(image_path_or_url)
        processed_image, mask = process_image(image)
        return extract_dominant_colors(processed_image, mask, color_count)[0]  # Return first dominant color
    except Exception as e:
        print(f"Error processing image {image_path_or_url}: {e}")
        return None


def extract_colors(images_list, color_count=2):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda img: process_single_image(img, color_count), images_list))
