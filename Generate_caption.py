import os
import pickle
import requests
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image

def load_model_from_path(model_path):
    model_link=os.path.abspath(model_path)
    if os.path.exists(model_link):
        try:
            model = load_model(model_link)
            print(f"Model from {model_link} loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model from {model_link}: {e}")
    else:
        print(f"File not found: {model_link}")
    return None

def tokenizer_load(path):
    with open(path, 'rb') as file:
         tokenizer = pickle.load(file)
    return tokenizer

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None


def extract_image_features_one(model, img_path):
    try:
        if img_path.startswith("http"):
            temp_path = "temp_image.jpg"
            img_path = download_image(img_path, temp_path)
            if img_path is None:
                return None

        if not os.path.exists(img_path):
            print(f"Error: Image path does not exist - {img_path}")
            return None

        image = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)

        if feature is None:
            print(f"Error: Model returned None for image - {img_path}")

        return feature
    except Exception as e:
        print(f"Exception in feature extraction: {e}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def idx_to_word(integer,tokenizer):
    for word ,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_captions(mapping):
    captions_list = []
    for key in mapping:
        captions_list.extend(mapping[key])
    return captions_list


def prepare_tokenizer(captions_list):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_list)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def calculate_max_length(captions_list):
    return max(len(caption.split()) for caption in captions_list)

def predict_caption(model, image, tokenizer, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word(yhat, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

def generate_caption(image_path,vgg16_model,model,tokenizer):
    features_image = extract_image_features_one(vgg16_model, image_path)
    if features_image is None:
       print("Error: No features extracted from the image.")
    y_pred = predict_caption(model, features_image, tokenizer, 18)
    return y_pred

