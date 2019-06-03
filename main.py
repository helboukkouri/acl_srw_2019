"""Main script"""
import os
import traceback
from copy import deepcopy
from helpers import set_seed
from config import default_config
from experiment import Experiment
#from notification.send_mail import notify


if __name__ == "__main__":

    print(f"\nProcess ID: {os.getpid()}\n")

    N_SEEDS = 1
    EXPERIMENT_NAME = 'chosen_experiment_name'

    # Parse args as default config
    CONFIG = default_config()

#    try:

    # List of all embedding strategies:
    BASE_EMBEDDINGS = [
        'word2vec_wiki',
        'elmo_small',
        ]

    EMBEDDING_CONCATENATIONS = [
        'word2vec_wiki+glove_wiki',
        ]

    CONCATENATIONS_WITH_ELMO = [
        'elmo_small+fasttext_wiki',
        ]

    MIXTURES_WITH_ELMO = [
        'elmo_small+fasttext_wiki+mixture',
        ]

    ALL_EMBEDDINGS = (
        BASE_EMBEDDINGS
        + EMBEDDING_CONCATENATIONS
        + CONCATENATIONS_WITH_ELMO
        + MIXTURES_WITH_ELMO)

    # Define a set of experiments
    if CONFIG.debug:
        EXPERIMENT_SEEDS = range(1)
        print(f'\n======================='
                + '\n    Debugging mode:'
                + '\n=======================\n')
    else:
        EXPERIMENT_SEEDS = range(N_SEEDS)

    print(f'Each experiment will be run with {len(EXPERIMENT_SEEDS)} different seeds.\n')

    print('These are the embedding strategies that will be tested:')
    for e in ALL_EMBEDDINGS:
        print(e)
    print('')

    # Experiment loop:
    for strategy in ALL_EMBEDDINGS:

        for seed in EXPERIMENT_SEEDS:

            # Set a seed for reproducibility
            set_seed(seed_value=seed)
            print(f'\nSeed: {seed + 1}/{len(EXPERIMENT_SEEDS)}.')

            experiment = Experiment(
                seed_value=seed,
                embedding_strategy=strategy, 
                name=EXPERIMENT_NAME,
                config=deepcopy(CONFIG))

            experiment.run()

#        if not CONFIG.debug:
#            notify(experiment_name=EXPERIMENT_NAME)

#    except:
#        if not CONFIG.debug:
#            TRACEBACK = traceback.format_exc()
#            notify(experiment_name=EXPERIMENT_NAME, log=TRACEBACK)
#        raise
