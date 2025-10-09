import os
import random

import nltk
import numpy as np
import pandas as pd
import regex as re
import tensorflow as tf
import tensorflow_hub as hub
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.applications import inception_v3, resnet50, vgg19, xception

# -------------------- Image Embeddings --------------------

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def initialize_image_models():
    """
    Initialize pre-trained CNN models from Keras Applications.

    Returns
    -------
    models : dict
        Dictionary with model names as keys and model instances as values.
    """
    models = {}
    model_configs = [
        ("VGG19", vgg19.VGG19),
        ("ResNet50", resnet50.ResNet50),
        ("Xception", xception.Xception),
        ("InceptionV3", inception_v3.InceptionV3),
    ]

    for name, model_class in model_configs:
        model = model_class(weights="imagenet", include_top=False, pooling="avg")
        model.trainable = False
        models[name] = model

    return models


def get_preprocessing_function(model_name):
    """
    Retrieve the appropriate preprocessing function for a given image model.

    Parameters
    ----------
    model_name : str
        Name of the image model.

    Returns
    -------
    function
        Corresponding preprocessing function.
    """
    preprocessing_functions = {
        "VGG19": vgg19.preprocess_input,
        "ResNet50": resnet50.preprocess_input,
        "Xception": xception.preprocess_input,
        "InceptionV3": inception_v3.preprocess_input,
    }

    if model_name not in preprocessing_functions:
        raise ValueError(f"Unknown model name: {model_name}")

    return preprocessing_functions[model_name]


def extract_image_features(images, model, preprocess_input_fn, batch_size=64):
    """
    Extract features from all images using a pre-trained model in batches.

    Parameters
    ----------
    images : dict
        Dictionary with ASINs as keys and image arrays as values.
    model : keras.Model
        Pre-trained CNN model for feature extraction.
    preprocess_input_fn : function
        Preprocessing function for the model.
    batch_size : int
        Number of images to process in each batch.

    Returns
    -------
    numpy.ndarray
        Array of extracted features (N, D).
    """
    image_ids = list(images.keys())
    num_images = len(image_ids)
    features = []

    for i in range(0, num_images, batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_images = np.vstack([images[asin] for asin in batch_ids])
        batch_images_preprocessed = preprocess_input_fn(batch_images.copy())
        batch_features = model.predict(batch_images_preprocessed)
        features.append(batch_features)

    return np.vstack(features)


# -------------------- Text Embeddings --------------------


def initialize_nltk():
    """
    Initialize NLTK data and global variables needed for text preprocessing.
    """
    required_nltk_data = [
        "punkt",
        "averaged_perceptron_tagger",
        "wordnet",
        "stopwords",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    for item in required_nltk_data:
        nltk.download(item)

    global stop_words, wl
    stop_words = set(stopwords.words("english"))
    wl = WordNetLemmatizer()


def get_wordnet_pos(pos_tag):
    """
    Map NLTK POS tag to WordNet POS tag for lemmatization.

    Parameters
    ----------
    pos_tag : str
        POS tag from NLTK's pos_tag.

    Returns
    -------
    str
        Corresponding WordNet POS tag.
    """
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return tag_map.get(pos_tag[0], wordnet.NOUN)


def preprocess_text(text):
    """
    Preprocess a single text string.

    Parameters
    ----------
    text : str
        The text to preprocess.

    Returns
    -------
    str
        The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [
        wl.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
    ]
    return " ".join(lemmatized_tokens)


def preprocess_texts(texts):
    """
    Preprocess a list of texts.
    """
    return [preprocess_text(text) for text in texts]


def initialize_models():
    """
    Initialize text embedding models.

    Returns
    -------
    dict
        Dictionary with model names as keys and models as values.
    """
    models = {
        "COUNT": CountVectorizer(stop_words="english", ngram_range=(1, 2)),
        "TFIDF": TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
    }

    models["USE"] = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    models["ST"] = SentenceTransformer("all-MiniLM-L6-v2")

    return models


def extract_text_features(texts, model_name, model):
    """
    Extract features from texts using the specified embedding model.

    Parameters
    ----------
    texts : list
        List of preprocessed texts.
    model_name : str
        Name of the embedding model.
    model : object
        The embedding model instance.

    Returns
    -------
    numpy.ndarray
        Array of extracted features.
    """
    if model_name in ["COUNT", "TFIDF"]:
        return model.transform(texts).toarray()
    elif model_name == "USE":
        return embed_texts_use(texts, model)
    elif model_name == "ST":
        return model.encode(texts, batch_size=32, show_progress_bar=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def embed_texts_use(texts, model_use, batch_size=64):
    """
    Embed texts using the Universal Sentence Encoder (USE).

    Parameters
    ----------
    texts : list
        List of preprocessed texts.
    model_use : tensorflow_hub.Module
        The USE model.
    batch_size : int
        Number of texts to process in each batch.

    Returns
    -------
    numpy.ndarray
        Array of embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model_use(batch_texts)
        embeddings.append(batch_embeddings.numpy())
    return np.vstack(embeddings)


# -------------------- Main Functions --------------------


def generate_image_embeddings(images, save_to_csv=False, dir_path=None):
    """Generates image embeddings for each model and saves them to disk.

    Parameters
    ----------
    images : dict
        Dictionary with ASINs as keys and preprocessed image arrays as values.
    file_path : str
        Path to save the embeddings.
    """

    models = initialize_image_models()

    # Extract features using each model
    features_dict = {}
    for model_name, model in models.items():
        preprocess_input_fn = get_preprocessing_function(model_name)
        features = extract_image_features(images, model, preprocess_input_fn)
        features_dict[model_name] = features

    # Save the embeddings to disk
    if save_to_csv and dir_path:
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        for model_name, features in features_dict.items():
            npy_file = os.path.join(dir_path, f"{model_name.lower()}_features.npy")
            np.save(npy_file, features)

            output_file = os.path.join(dir_path, f"{model_name.lower()}_features.csv")
            features_df = pd.DataFrame(features)
            features_df.insert(0, "asin", images.keys())
            features_df.to_csv(output_file, index=False, float_format="%.18e")

        print(f"Saved image embeddings to {dir_path}.")

    return features_dict


def generate_text_embeddings(texts, save_to_csv=False, dir_path=None):
    """Generates text embeddings for each model and saves them to disk.

    Parameters
    ----------
    texts : dict
        Dictionary with ASINs as keys and text data as a dictionary as values.
    file_path : str
        Path to save the embeddings.
    """

    initialize_nltk()

    # Get text sources from secondary key of texts
    text_sources = set(key for asin_dict in texts.values() for key in asin_dict.keys())
    review_keys = sorted(key for key in text_sources if key.startswith("user_review_"))

    # Preprocess texts for each source
    texts_per_source = {}
    for source in text_sources:
        text = [texts[asin].get(source, "") for asin in texts.keys()]
        texts_per_source[source] = preprocess_texts(text)

    # Combine texts for vectorizer fitting
    all_preprocessed_texts = []
    for text in texts_per_source.values():
        all_preprocessed_texts.extend(text)

    # Initialize models
    embedding_models = initialize_models()

    # Fit COUNT and TFIDF vectorizers
    for model_name in ["COUNT", "TFIDF"]:
        try:
            embedding_models[model_name].fit(all_preprocessed_texts)
        except Exception as e:
            print(texts)
            print(f"Error fitting {model_name}: {e}")
            print(f"Directory: {dir_path}")

    # Extract features for each text source and model
    features = {}
    for source in text_sources:
        features[source] = {}
        for model_name, model in embedding_models.items():
            features[source][model_name] = extract_text_features(
                texts_per_source[source], model_name, model
            )

    if review_keys:
        # Average user review embeddings
        features["user_reviews"] = {}
        for model_name in embedding_models:
            review_embeddings = [
                features[review_key][model_name] for review_key in review_keys
            ]
            features["user_reviews"][model_name] = np.mean(
                np.stack(review_embeddings), axis=0
            )

        # Remove individual review features
        for review_key in review_keys:
            features.pop(review_key)

    # Save the embeddings to disk
    if save_to_csv and dir_path:
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        for source, features_dict in features.items():
            for model_name, features in features_dict.items():
                output_file = os.path.join(
                    dir_path, f"{source}_{model_name}_features.npy"
                )
                np.save(output_file, features)

                output_file = os.path.join(
                    dir_path, f"{source}_{model_name}_features.csv"
                )
                features_df = pd.DataFrame(features)
                features_df.insert(0, "asin", texts.keys())
                features_df.to_csv(output_file, index=False, float_format="%.18e")

        print(f"Saved text embeddings to {dir_path}.")

    return features
