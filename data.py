"""AllenNLP modules for loading concept detection & assertion classification data."""
import os
import re
import html
import random

from math import log10, floor
from functools import reduce
from typing import Iterator, List, Dict

from overrides import overrides

from allennlp.data import DatasetReader, TokenIndexer, Token, Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from nltk.tokenize import RegexpTokenizer
from config import I2B2_PATH

TOKENIZER = RegexpTokenizer(r'\'s|\w+[:/-]\w+[:/-]*\w*|\w+|[^\w\s]+')


def apply_new_tokenization(tokens, labels):

    assert len(tokens) == len(labels)
    if len(tokens) == 0:
        raise ValueError('Token sequence is empty.')
        
    new_tokens, new_labels = [], []

    for token, label in zip(tokens, labels):

        retokenized_token = TOKENIZER.tokenize(token)
        if retokenized_token != [token]:
            if label != 'O':

                label_pos = label[:2]
                label_type = label.split('-')[-1]
                if label_pos == 'B-':
                    new_label = [label] + (len(retokenized_token) - 1) * ['I-' + label_type]
                elif label_pos == 'I-':
                    new_label = [label] * len(retokenized_token)
            else:
                new_label = [label] * len(retokenized_token)

            new_tokens.extend(retokenized_token)
            new_labels.extend(new_label)

        else:
            new_tokens.append(token)
            new_labels.append(label)

    if new_tokens:
        return new_tokens, new_labels
    else:
        return [''], ['O']


class ConceptDatasetReader(DatasetReader):
    """
    DatasetReader for Concept detection data.
    """
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = {'tokens': SingleIdTokenIndexer()}) -> None:

        super().__init__(lazy=False)
        self.token_indexers = token_indexers

    @overrides
    def text_to_instance(
            self,
            tokens: List[Token],
            labels: List[str]) -> Instance:
        """Converts a sequence of Tokens to an Instance."""

        tokens_field = TextField(
            tokens=tokens,
            token_indexers=self.token_indexers)

        labels_field = SequenceLabelField(
            labels=labels,
            sequence_field=tokens_field)

        fields = {"token_sequence": tokens_field, "label_sequence": labels_field}
        return Instance(fields)

    @overrides
    def read(
            self,
            is_training: bool) -> Iterator[Instance]:
        """Yields an Instance from each train/test token sequence."""

        instances = []
        for tokens, labels in zip(*load_concept_data(is_training)):

            # Preprocessing
            text = ' '.join(tokens)
            text = html.unescape(text)
            text = re.sub(r'\d+', to_most_significant_digit, text)
            tokens = text.split(' ')
            # Uncomment and adapt to use a custom tokenization
            #tokens, labels = apply_new_tokenization(tokens, labels)

            instance = self.text_to_instance(
                tokens=[Token(token) for token in tokens], labels=labels)
            instances.append(instance)

        # Split train into actual train and validation set
        if is_training:
            random.shuffle(instances)
            instances = (instances[:-5000], instances[-5000:])

        return instances


def to_most_significant_digit(match):
    """Converts a number to its most significant number version.
    Ex: 0.3352 -> 0.3. The normal formula would be: floor(x / 10**exp) * 10**exp.
    But because of float approximations in Python this can yield 0.3000...01 instead of 0.3
    So this function uses a workaround.
    """
    x = match.group()
    length = len(x)
    x = float(x)
    if x == 0:
        result = '0'
    else:
        if x < 0:
            sign = -1
            x *= -1
        else:
            sign = 1

        exp = floor(log10(x))
        if exp == 0:
            coef = 1
        elif exp > 0:
            coef = reduce(lambda x, y: x * y, [1] + [10] * exp)
        elif exp < 0:
            exp *= -1
            coef = reduce(lambda x, y: x / y, [1] + [10] * exp)
        result = str(sign * floor(x / coef) * coef)

    if len(result) < length:
        result = '0' * (length - len(result)) + result
    return result


def load_concept_data(is_training):
    # Whether load the data for the training or test step:
    step = 'train' if is_training else 'test'

    # Paths of clinical records and their annotations:
    base_path = os.path.join(I2B2_PATH, f'2010/{step}/')
    base_txt_path = base_path + 'texts/'
    base_con_path = base_path + 'concepts/'

    # Names of clinical record files in alphabetical order:
    txt_filenames = os.listdir(base_txt_path)
    txt_filenames.sort()
    print(f'Number of {step} files: {len(txt_filenames)}')

    #######################################################################################
    #        Populate a list of token sequences and their IOB tags / each document
    #######################################################################################

    all_token_sequences = []
    all_iob_sequences = []
    count_concepts = 0
    for filename in txt_filenames:

        # Read text & annotations files:
        text = open(base_txt_path + filename, 'r', encoding='utf-8-sig').read().lower()
        annotations = open(base_con_path + filename[:-3] + 'con', 'r', encoding='utf-8-sig').read()

        # Convert text to token sequences:
        token_sequences = [re.split(r'\ +', sentence) for sentence in text.split('\n')]

        # Initialize all tags to 'O':
        iob_sequences = [['O'] * len(seq) for seq in token_sequences]

        # Convert annotations to a list of concepts & skip files with no concepts:
        if annotations == '':
            all_token_sequences.extend(token_sequences)
            all_iob_sequences.extend(iob_sequences)
            continue
        else:
            annotations = annotations.strip().split('\n')

        for annotation in annotations:

            # Extract each concept along with its type and span:
            concept_name = re.findall(r'c="(.*?)" \d', annotation)[0]
            concept_tag = re.findall(r't="(.*?)"$', annotation)[0]
            concept_span_string = re.findall(r'(\d+:\d+\ \d+:\d+)', annotation)[0]

            # Convert span to integers
            span_1, span_2 = concept_span_string.split(' ')
            (line1, start), (line2, end) = span_1.split(':'), span_2.split(':')
            line1, line2, start, end = int(line1), int(line2), int(start), int(end)

            # Check that no concept spans over two lines (token sequences):
            assert line1 == line2

            # Check that the concept is at the correct position in the document:
            concept_name = re.sub(r'\ +', ' ', concept_name)
            original_text = ' '.join(token_sequences[line1 - 1][start:end + 1])
            if concept_name != original_text.lower():
                print(concept_name, original_text, annotation)
                raise RuntimeError('Can\'t find the concept where it should be')

            # Tag the first token of the concept as a B, then the others as I:
            first = True
            for start_id in range(start, end + 1):
                if first:
                    iob_sequences[line1 - 1][start_id] = 'B-' + concept_tag
                    count_concepts += 1
                    first = False
                else:
                    iob_sequences[line1 - 1][start_id] = 'I-' + concept_tag

        all_token_sequences.extend(token_sequences)
        all_iob_sequences.extend(iob_sequences)

    print(f'Number of {step} concepts: {count_concepts}')
    return all_token_sequences, all_iob_sequences


if __name__ == "__main__":

    # A single training instance:
    train, valid = ConceptDatasetReader().read(is_training=True)
    for i in train:
        print(i)
        break

    # A single test instance:
    for i in ConceptDatasetReader().read(is_training=False):
        print(i)
        break
