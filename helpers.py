"""This module implements some helper functions."""
import os
import random
import logging

import torch
import numpy as np
from gensim.models import KeyedVectors
from fastText import load_model as load_fastText_model


def set_seed(seed_value):
    """This function sets a seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(save_dir):
    """Returns a logger that writes both in stdout and in a logfile.
    """
    logging_level = logging.INFO
    logger = logging.getLogger()
    # Reset the logger to avoid duplicates
    list(map(logger.removeHandler, logger.handlers[:]))
    list(map(logger.removeFilter, logger.filters[:]))

    logger.setLevel(logging_level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(save_dir, "experiment.log"))
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class GloveEmbeddings():
    def __init__(self, path):
        f = open(path,'r', encoding='utf-8')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        self.model = model
    
    def get_vector(self, token):
        return self.model[token]


class FastTextEmbeddings():
    def __init__(self, path):
        self.model = load_fastText_model(path)
    
    def get_vector(self, token):
        return self.model.get_word_vector(token)


def load_pretrained_embeddings(vocabulary, embedding, config, logger):

    source, name = embedding.split('_')
    if source == 'word2vec':
        embedding_dir = os.path.join('embeddings/word2vec/pretrained', name)
        embedding_path = os.path.join(embedding_dir, "vectors.bin")
        embedding_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    
    elif source == 'glove':
        embedding_dir = os.path.join('embeddings/glove/pretrained', name)
        embedding_path = os.path.join(embedding_dir, "vectors.txt")
        embedding_model = GloveEmbeddings(embedding_path)
    
    elif source == 'fasttext':
        embedding_dir = os.path.join('embeddings/fasttext/pretrained', name)
        embedding_path = os.path.join(embedding_dir, "vectors.bin")
        embedding_model = FastTextEmbeddings(embedding_path)
    
    logger.info("Using pre-trained embeddings from:\n{}\n".format(embedding_path))

    vocabulary_size = vocabulary.get_vocab_size()
    embedding_matrix = np.random.normal(size=(vocabulary_size, config.embedding_dim))

    OOV_count = 0
    for i in range(vocabulary_size):
        token = vocabulary.get_token_from_index(i)
        try:
            embedding_matrix[i] = embedding_model.get_vector(token)
        except KeyError:
            OOV_count += 1
    logger.info(f"Number of OOV in {embedding}: {OOV_count}")
    return embedding_matrix
