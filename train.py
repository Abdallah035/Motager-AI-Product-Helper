import os
import random
import numpy as np
from tqdm import tqdm
from Generate_caption import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, GRU, add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def extract_image_features(model, image_folder):
    features = {}
    directory = os.path( image_folder)
    for item in tqdm(os.listdir(directory), desc="Extracting Features"):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            try:
                image = load_img(item_path, target_size=(224, 224))
                img_array = img_to_array(image)
                img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
                img_array = preprocess_input(img_array)
                feature = model.predict(img_array, verbose=0)
                image_id = item.split('.')[0]
                features[image_id] = feature
            except Exception as e:
                print(f"Error processing image {item_path}: {e}")
    return features


def read_captions_file(file_path):
    try:
        with open(file_path, 'r') as file:
            next(file)
            captions = file.read()
        return captions
    except Exception as e:
        raise RuntimeError(f"Error reading the file: {e}")


def create_image_caption_mapping(captions):
    mapping = {}
    for line in tqdm(captions.split('\n'), desc="Processing Captions"):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping


def preprocess_text(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.replace('[^A-Za-z]', ' ').replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


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


def split(image_ids, train_ratio, val_ratio=None):
    random.shuffle(image_ids)
    total = len(image_ids)
    train_split = int(total * train_ratio)
    val_split = int(total * (train_ratio + val_ratio)) if val_ratio else train_split
    train_ids = image_ids[:train_split]
    val_ids = image_ids[train_split:val_split] if val_ratio else []
    test_ids = image_ids[val_split:]
    return train_ids, val_ids, test_ids


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
                X1, X2, y = [], [], []
                n = 0


def build_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)

    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = GRU(256, recurrent_dropout=0.3, return_sequences=False)(se2)

    decoder1 = add([fe3, se3])
    decoder2 = LayerNormalization()(decoder1)
    decoder3 = Dense(512, activation='relu')(decoder2)
    decoder4 = Dropout(0.3)(decoder3)
    outputs = Dense(vocab_size, activation='softmax')(decoder4)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def load_existing_or_new_model(vocab_size, max_length, model_path="seven_version_model.keras"):
    if os.path.exists(model_path):
        print("Loading existing model...")
        return load_model(model_path)
    else:
        print("No existing model found. Creating a new one...")
        return build_model(vocab_size, max_length)


def continue_training(model, train, val, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    steps = len(train) // batch_size

    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        validation_generator = data_generator(val, mapping, features, tokenizer, max_length, vocab_size, batch_size)

        model.fit(generator, validation_data=validation_generator, epochs=1, steps_per_epoch=steps,
                  validation_steps=len(val) // batch_size, verbose=1, callbacks=[early_stopping, lr_scheduler])

    model.save("seven_version_model.keras")
    print("Updated model saved successfully.")

#
# model = load_existing_or_new_model(vocab_size, max_length)
# continue_training(model, train_ids, val_ids, mapping, features, tokenizer, max_length, vocab_size, batch_size=64,
#                   epochs=10)
