"""Global variables and default config with argpase."""
import argparse

I2B2_PATH = "i2b2_data/"
# This should point to a folder that is set up like this:
# i2b2_data/
# |
#  --> 2010/
#      |
#      | --> test/
#      |     |
#      |     | --> concepts/
#      |     |     |
#      |     |      --> *.con
#      |     |
#      |       --> texts/
#      |           |
#      |            --> *.txt
#        --> train/
#            |
#            | --> concepts/
#            |     |
#            |      --> *.con
#            |
#              --> texts/
#                  |
#                   --> *.txt


def str2bool(value):
    """Convert strings to the corresponding booleans."""
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def default_config():
    """Returns a config objects with many important parameters for the experiment."""

    example_use = """
    - To use with GPU:
    python main.py --device='gpu'
    - Or with CPU only:
    python main.py --device='cpu'
    """

    parser = argparse.ArgumentParser(
        epilog=example_use,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Trains a model for the i2b2/VA 2010 Concept Detection Task.')

    # Training config
    parser.add_argument(
        "--device", type=str, required=True,
        help="Which kind of device to use. Ex: \"cpu\" or \"gpu\".")
    parser.add_argument(
        "--n_epochs", type=int, default=100,
        help="Number of training epochs.")
    parser.add_argument(
        "--batch_size_train", type=int, default=32,
        help="Batch size for training and validation.")
    parser.add_argument(
        "--batch_size_test", type=int, default=256,
        help="Batch size for testing.")

    # Embedding config
    parser.add_argument(
        "--embedding_dim", type=int, default=256,
        help="Dimension of the word embeddings.")
    parser.add_argument(
        "--embeddings_are_trainable", type=str2bool, default=True,
        help="The number of most frequent tokens to keep in the vocabulary.")

    # Model config
    parser.add_argument(
        "--hidden_dim", type=int, default=256,
        help="Number of LSTM units.")
    parser.add_argument(
        "--dropout_rate", type=float, default=0.5,
        help="Dropout rate for the LSTM.")
    parser.add_argument(
        "--n_lstm_layers", type=int, default=3,
        help="Number of recurrent layers.")
    parser.add_argument(
        "--bidirectional", type=str2bool, default=True,
        help="Whether to use bi-directional LSTMs.")
    parser.add_argument(
        "--use_crf", type=str2bool, default=True,
        help="Whether to use a CRF layer for decoding.")

    # Debug config
    parser.add_argument(
        "--debug", type=str2bool, default=False,
        help="Set on/off debug mode.")

    # Set number of epochs to 1 during debugging
    config = parser.parse_args()
    if config.debug:
        vars(config).update({'n_epochs': 1})

    return config
