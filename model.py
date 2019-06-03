"""Define models for sequence labeling."""
import argparse
from typing import Dict
from overrides import overrides

import torch
from torch import nn

from custom.custom_crf import CrfTagger
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


class SequenceLabelingModel(Model):
    def __init__(self,
                 embedders: nn.ModuleList,
                 encoder: Seq2SeqEncoder,
                 vocabulary: Vocabulary,
                 config: argparse.Namespace) -> None:

        super().__init__(vocabulary)

        self.config = config
        self.embedders = embedders
        self.encoder = encoder

        self.linear_layer = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocabulary.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()
        self.F1 = SpanBasedF1Measure(vocabulary, tag_namespace='labels')

    def forward(self,
                token_sequence: Dict[str, torch.Tensor],
                label_sequence: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(token_sequence)

        if 'mixture' in self.config.embedding_strategy:
            word2vec_embedder = self.embedders[1]

            # Keep in mind to manage any custom models that don't produce 256-dim vectors
            if any([strat in self.config.embedding_strategy
                    for strat in ['elmo_original', 'elmo_pubmed']]):
                word2vec_embeddings = word2vec_embedder(
                    {'tokens': token_sequence['tokens']})
                word2vec_embeddings = torch.cat([
                    word2vec_embeddings,
                    word2vec_embeddings,
                    word2vec_embeddings,
                    word2vec_embeddings], dim=2)
            else:
                word2vec_embeddings = word2vec_embedder(
                    {'tokens': token_sequence['tokens']})

            # Pad with zeros at BOS and EOS
            batch_size, _, embedding_dim = word2vec_embeddings.shape
            zeros = torch.zeros([batch_size, 1, embedding_dim])
            if self.config.device == 'gpu':
                zeros = zeros.cuda()

            padded_word2vec_embeddings = torch.cat(
                [zeros, word2vec_embeddings, zeros], dim=1)

            elmo_embedder = self.embedders[0]
            embeddings = elmo_embedder(
                {'characters': token_sequence['characters']},
                word2vec_embeddings=padded_word2vec_embeddings)

        else:
            embeddings = []
            for embedder in self.embedders:
                if hasattr(embedder, 'token_embedder_characters'):
                    embeddings.append(
                        embedder({'characters': token_sequence['characters']}))
                elif hasattr(embedder, 'token_embedder_tokens'):
                    embeddings.append(
                        embedder({'tokens': token_sequence['tokens']}))

            embeddings = torch.cat(embeddings, dim=2)

        encoder_output = self.encoder(embeddings, mask)

        label_logits = self.linear_layer(encoder_output)

        self.accuracy(label_logits, label_sequence, mask)
        self.F1(label_logits, label_sequence, mask)

        output = {
            "label_logits": label_logits,
            "loss": sequence_cross_entropy_with_logits(label_logits, label_sequence, mask)}

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {"accuracy": self.accuracy.get_metric(reset)}

        for name, value in self.F1.get_metric(reset).items():
            if 'overall' in name: metrics[name] = value

        return metrics


class SequenceLabelingModelWithCRF(CrfTagger):
    def __init__(self,
                 embedders: nn.ModuleList,
                 encoder: Seq2SeqEncoder,
                 vocabulary: Vocabulary,
                 config: argparse.Namespace) -> None:

        super().__init__(vocab=vocabulary,
                         text_field_embedder=embedders,
                         encoder=encoder)

        self.config = config
        self.embedders = self.text_field_embedder

        self.linear_layer = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocabulary.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()
        self.F1 = SpanBasedF1Measure(vocabulary, tag_namespace='labels')


    @overrides
    def forward(self,
                token_sequence: Dict[str, torch.Tensor],
                label_sequence: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        mask = get_text_field_mask(token_sequence)

        if 'mixture' in self.config.embedding_strategy:
            word2vec_embedder = self.embedders[1]

            # Keep in mind to manage any custom models that don't produce 256-dim vectors
            if any([strat in self.config.embedding_strategy
                    for strat in ['elmo_original', 'elmo_pubmed']]):
                word2vec_embeddings = word2vec_embedder(
                    {'tokens': token_sequence['tokens']})
                word2vec_embeddings = torch.cat([
                    word2vec_embeddings,
                    word2vec_embeddings,
                    word2vec_embeddings,
                    word2vec_embeddings], dim=2)
            else:
                word2vec_embeddings = word2vec_embedder(
                    {'tokens': token_sequence['tokens']})

            # Pad with zeros at BOS and EOS
            batch_size, _, embedding_dim = word2vec_embeddings.shape
            zeros = torch.zeros([batch_size, 1, embedding_dim])
            if self.config.device == 'gpu':
                zeros = zeros.cuda()

            padded_word2vec_embeddings = torch.cat(
                [zeros, word2vec_embeddings, zeros], dim=1)

            elmo_embedder = self.embedders[0]
            embeddings = elmo_embedder(
                {'characters': token_sequence['characters']},
                word2vec_embeddings=padded_word2vec_embeddings)

        else:
            embeddings = []
            for embedder in self.embedders:
                if hasattr(embedder, 'token_embedder_characters'):
                    embeddings.append(
                        embedder({'characters': token_sequence['characters']}))
                elif hasattr(embedder, 'token_embedder_tokens'):
                    embeddings.append(
                        embedder({'tokens': token_sequence['tokens']}))

            embeddings = torch.cat(embeddings, dim=2)

        encoded_text = self.encoder(embeddings, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"label_logits": logits, "mask": mask, "tags": predicted_tags}

        # Add negative log-likelihood as loss
        log_likelihood = self.crf(logits, label_sequence, mask)
        output["loss"] = -log_likelihood

        # Represent viterbi tags as "class probabilities" that we can
        # feed into the metrics
        class_probabilities = logits * 0.
        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1

        self.accuracy(logits, label_sequence, mask)
        self.F1(logits, label_sequence, mask)

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {"accuracy": self.accuracy.get_metric(reset)}

        for name, value in self.F1.get_metric(reset).items():
            if 'overall' in name: metrics[name] = value

        return metrics
