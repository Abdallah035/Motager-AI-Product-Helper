import os
import pickle

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
            # print(f"Model from {model_link} loaded successfully!")
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
def extract_image_features_one(model, img_path):
    try:
        # Preprocess the image
        image = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)

        # Extract and return the feature
        feature = model.predict(img_array, verbose=0)
        return feature
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

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
        # Add start tag for generation process
        in_text = 'startseq'

        # Iterate over the max length of sequence
        for i in range(max_length):
            # Encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # Pad the sequence
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            # Predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # Get index with high probability
            yhat = np.argmax(yhat)
            # Convert index to word
            word = idx_to_word(yhat, tokenizer)

            # Stop if word not found
            if word is None:
                break

            # Append word as input for generating the next word
            in_text += " " + word

            # Stop if we reach end tag
            if word == 'endseq':
                break

        return in_text

def generate_caption(image_path):
    #load the vgg16_model
    vgg16_model = load_model_from_path('models/vgg16_feature_extractor.keras')
    # Extract features from the image using the feature extraction function
    features_image = extract_image_features_one(vgg16_model, image_path)
    if features_image is None:
       print("Error: No features extracted from the image.")
    #load fifth_version_model
    fifth_version_model = load_model_from_path('models/fifth_version_model.keras')
    #load tokenizer
    tokenizer = tokenizer_load('models/tokenizer.pkl')
    # Predict the caption
    y_pred = predict_caption(fifth_version_model, features_image, tokenizer, 18)
    return y_pred

