"""Module that defines experiments."""
import os
import json
import time
import requests
import numpy as np

import torch
from torch.nn import LSTM, ModuleList
from torch.optim import Adam

from allennlp.data import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from custom.custom_elmo import ElmoTokenEmbedder as CustomElmoTokenEmbedder
from custom.custom_embedder import CustomTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from evaluation import i2b2_evaluation
from helpers import set_logger, load_pretrained_embeddings
from model import SequenceLabelingModel, SequenceLabelingModelWithCRF
from data import ConceptDatasetReader


def run_experiment(config, save_dir, logger, experiment_name):
    """Runs the experiment."""

    ####################################################################################
    #                               Load and index the data
    ####################################################################################

    # Set the token indexers depending on whether we are using ELMo or not:
    token_indexers = {}
    if 'elmo' not in config.embedding_strategy:
        token_indexers['tokens'] = SingleIdTokenIndexer()
    elif all(['elmo' in strategy for strategy in config.embedding_strategy.split('+')]):
        token_indexers['characters'] = ELMoTokenCharactersIndexer()
    else:
        token_indexers['tokens'] = SingleIdTokenIndexer()
        token_indexers['characters'] = ELMoTokenCharactersIndexer()

    # Build a dataset reader that loads the data and indexes it:
    Reader = ConceptDatasetReader
    
    training_dataset, validation_dataset = Reader(
        token_indexers=token_indexers).read(is_training=True)
    
    test_dataset = Reader(
        token_indexers=token_indexers).read(is_training=False)

    ####################################################################################
    #                              Define the task vocabulary
    ####################################################################################

    # Build a vocabulary from the train/val/test datasets:
    vocabulary = Vocabulary().from_instances(
        training_dataset + validation_dataset + test_dataset,
        min_count={'tokens': 5})

    vocabulary_size = vocabulary.get_vocab_size(namespace='tokens')
    logger.info(f"Number of tokens in the vocabulary: {vocabulary_size}")

    # Build a generator that yields batches of token indices:
    iterator = BucketIterator(
        sorting_keys=[("token_sequence", "num_tokens")],
        batch_size=config.batch_size_train)
    iterator.index_with(vocabulary)

    ####################################################################################
    #                                Define the embedders
    ####################################################################################

    embedders = []
    final_embedding_dim = 0
    
    for embedding in config.embedding_strategy.split('+'):

        if embedding == 'mixture':  # Ignore this since it's not really an embedding
            continue
        
        # Manage cases where embeddings have dimensions != config.embedding_dim
        if ('elmo_original' in embedding) or ('elmo_pubmed' in embedding):
            final_embedding_dim += 1024
        else:
            final_embedding_dim += config.embedding_dim

        ####################################################################################
        #                                      ELMo
        ####################################################################################

        if 'elmo' in embedding:
            # This downloads a model from AllenNLP website: https://allennlp.org/elmo
            # To avoid downloading every single time, you can point to it locally.

            if embedding == 'elmo_small':
                weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                               '2x1024_128_2048cnn_1xhighway/'
                               'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
                options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                                '2x1024_128_2048cnn_1xhighway/'
                                'elmo_2x1024_128_2048cnn_1xhighway_options.json')
            elif embedding == 'elmo_original':
                weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                                '2x4096_512_2048cnn_2xhighway/'
                                'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
                options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                                '2x4096_512_2048cnn_2xhighway/'
                                'elmo_2x4096_512_2048cnn_2xhighway_options.json')
            elif embedding == 'elmo_pubmed':
                weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                               'contributed/pubmed/'
                               'elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5')
                options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                                'contributed/pubmed/'
                                'elmo_2x4096_512_2048cnn_2xhighway_options.json')

            for _ in range(10):  # Try to load elmo, max retries = 10
                try:
                    if 'mixture' in config.embedding_strategy:
                        elmo_embedding = CustomElmoTokenEmbedder(options_file, weight_file)
                        embedder = CustomTextFieldEmbedder({"characters": elmo_embedding})
                    else:
                        elmo_embedding = ElmoTokenEmbedder(options_file, weight_file)
                        embedder = BasicTextFieldEmbedder({"characters": elmo_embedding})
                except requests.exceptions.ConnectionError:
                    logger.info(
                        'ConnectionError while downloading ELMo weights. Retrying in 5 minutes...')
                    time.sleep(seconds=300)  # wait 5 minutes
                    continue
                break

            logger.info(f'\n\nUsing contextualized word embeddings: ELMo ({embedding}).\n')

        ####################################################################################
        #                                     Word2vec
        ####################################################################################

        else:
            embedding_layer = Embedding(
                num_embeddings=vocabulary_size,
                embedding_dim=config.embedding_dim)

            if embedding != 'random':
                embedding_matrix = load_pretrained_embeddings(
                    vocabulary=vocabulary, config=config, embedding=embedding, logger=logger)
                embedding_layer.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})

            # Trainable embeddings ?
            embedding_layer.weight.requires_grad = config.embeddings_are_trainable
            if embedding_layer.weight.requires_grad:
                logger.info(f'{embedding} embeddings are trainable.')
            else:
                logger.info(f'{embedding} embeddings are non-trainable.')

            embedder = BasicTextFieldEmbedder({"tokens": embedding_layer})

        embedders.append(embedder)

    # Make a PyTorch Module List from all these embedders
    embedders = ModuleList(embedders)
    if 'mixture' in config.embedding_strategy:
        # ELMo (Original) and ELMo (PubMed) produce 1024-dimensional embeddings
        if ('elmo_original' in config.embedding_strategy) or ('elmo_pubmed' in config.embedding_strategy):
            final_embedding_dim = 1024
        else:
            final_embedding_dim = config.embedding_dim

    ####################################################################################
    #                                 Define the model
    ####################################################################################

    lstm_layer = LSTM(
        input_size=final_embedding_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.n_lstm_layers,
        dropout=config.dropout_rate,
        bidirectional=config.bidirectional,
        batch_first=True)

    encoder = PytorchSeq2SeqWrapper(module=lstm_layer)

    if config.use_crf:
        model = SequenceLabelingModelWithCRF(
            embedders=embedders, encoder=encoder, vocabulary=vocabulary, config=config)
    else:
        model = SequenceLabelingModel(
            embedders=embedders, encoder=encoder, vocabulary=vocabulary, config=config)

    logger.info(f'\n\nUsing the following model:\n\n {model}\n')

    optimizer = Adam(model.parameters())

    ####################################################################################
    #                                   Train the model
    ####################################################################################

    # Using validation span F1 for early stopping
    validation_metrics = '+f1-measure-overall'

    trainer = Trainer(
        model=model,
        validation_metric=validation_metrics,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset, patience=3,
        num_epochs=config.n_epochs,
        cuda_device=-1 if config.device == 'cpu' else 0)

    # You can interupt training at any time by pressing Ctrl+C
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass

    ####################################################################################
    #                            Evaluate the final model
    ####################################################################################

    # the evaluation .jar doesn't support '-' characters
    prediction_dir = experiment_name.replace('-', '_') + '/'

    # Reduce the batch size to avoid OOM errors for elmo:
    batch_size_test = 32 if 'elmo' in config.embedding_strategy else config.batch_size_test

    # Batches of test data:
    test_batches = [
        test_dataset[batch_size_test * i:batch_size_test * (i + 1)]
        for i in range(1 + len(test_dataset) // batch_size_test)]

    # Compute model predictions:
    predictions = []
    for batch in test_batches:
        pred = model.forward_on_instances(batch)
        predictions.extend(pred)

    # Post-process predictions to use the official i2b2 evaluation script:    
    if config.use_crf:
        predictions = [p['tags'] for p in predictions]
    else:
        predictions = [
            [vocabulary.get_token_from_index(i, namespace='labels')
            for i in p['label_logits'].argmax(-1)]
            for p in predictions]

    lengths = [len(instance['token_sequence']) for instance in test_dataset]
    predictions = [p[:l] for p, l in zip(predictions, lengths)]

    logger.info(f'Using {prediction_dir} as a temporary prediction directory.')
    i2b2_evaluation(predictions,
        output_path=prediction_dir,
        logger=logger,
        sort_files=False,
# Uncomment and point to a gold standard using a custom tokenization if needed
#        i2b2_path=I2B2_RETOKENIZED_PATH
        )

    ####################################################################################
    #                            Save the model/vocabulary
    ####################################################################################

    # Uncomment these lines to save the final model   
#    os.makedirs(os.path.join(save_dir, 'model'))
#    with open(os.path.join(save_dir, 'model', 'final_model.th'), 'wb') as f:
#        torch.save(model.state_dict(), f)
#
#    vocabulary.save_to_files(os.path.join(save_dir, 'model', 'vocabulary'))


####################################################################################
#                             Main "Experiment" Class
####################################################################################

class Experiment():
    """A class that runs an experiment according to a given set of config parameters."""
    def __init__(self, name, config, **params_to_override):

        self.name = name
        self.start_time = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')

        # Update the default config with any input keyword args
        self.config = config
        vars(self.config).update(params_to_override)

        # Create a directory for saving the expriment's results
        self.make_experiment_directory()

        # A logger that writes to both stdout and a logfile
        self.logger = set_logger(self.save_dir)

        # Save configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)

    def make_experiment_directory(self):
        """Creates a directory for saving the experiment's models and results."""
        self.save_dir = os.path.join(
            "results", self.name,
            self.config.embedding_strategy, self.start_time)

        # Prepend 'debug' when in debug mode
        if self.config.debug:
            self.save_dir = os.path.join("debug", self.save_dir)

        # Create directory
        os.makedirs(self.save_dir)

    def run(self):
        """Runs the experiment."""
        run_experiment(
            self.config, self.save_dir,
            self.logger, self.name + '_' + self.start_time)
