# Embedding Strategies for Specialized Domains: Application to Clinical Entity Recognition

## Python environment

The code was tested on Linux (Ubuntu 16.04.4 LTS) and Python (3.6). Using Anaconda, install a new environment from the `.yml` file:

`conda env create --name ACL_PAPER_env -f=environment.yml`

Then activate it:

`source activate ACL_PAPER_env`

## Steps for reproducing the experiments

### Step 1: Prepare your corpora
Prepare each corpus you want to train a set of static embeddings on by adding a folder `embeddings/corpora/{corpus_name}/` where `{corpus_name}` is the name of your corpus. Then put your **preprocessed corpus** as a single text file called `corpus.txt` inside that folder.

The code comes with a small example corpus from the English Wikipedia in `embeddings/corpora/wiki/`. This is only there as an example and will not result in great performance.

### Step 2: Download & compile embedding codes
Inside the `embeddings` folder you will also find three other folders called `word2vec`, `glove` and `fasttext`. Each of these folders contrain a shell script called `download_and_compile_code.sh`. You can run this script to download the method's source code and compile it in preparation for the embedding training.

`bash download_and_compile_code.sh`

### Step 3: Train embeddings on your corpus
The `word2vec`, `glove` and `fasttext` also include another script called `train_embeddings.sh`. By default, runninng this script will result in training an embedding with the same parameters as in the paper, using the method of your choice, on each corpus available inside `embeddings/corpura/`.

`bash train_embeddings.sh`

### Step 4: Prepare experiments
Edit the `main.py` script by changing the list of models to run, as well as the name of the experiment. This name will be used to create a results folder in `results/{experiment_name}`. Changing the experiment name will result in a different folder - this can be used to group your results as you wish.

To use an embedding trained using method `{method}` and corpus `{corpus}` use the name `{method}_{corpus}`.
Also, there are three available ELMo embeddings: Small, Original and PubMed as described in https://allennlp.org/elmo. To use them use the names: `elmo_small`, `elmo_original` and `elmo_pubmed`.

For everything related to combining ELMo with other embeddings (word2vec, fastText and GloVe), you can refer to the examples in `main.py` script.

### Step 5: Run the experiments
Edit the `run_experiment.sh` script and chose a GPU if available. If no GPUs are available, change the parameter `--device='gpu'` to `--device='cpu'` then run the script.

`bash run_experiment.sh`

This should run your experiments in "debug mode". This means that each model will be trained for only one epoch. This mode can be used to check that everything is working fine before running bigger trainings. To run a definitive training, change the parameter `--debug=True` to `--debug=False`.
