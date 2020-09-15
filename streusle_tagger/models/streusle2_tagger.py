from typing import Any, Dict, List, Optional, Tuple
import logging

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
import allennlp.nn.util as util
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("streusle2_tagger")
class Streusle2Tagger(Model):
    """
    The ``Streusle2Tagger`` embeds a sequence of text before (optionally)
    encoding it with a ``Seq2SeqEncoder`` and passing it through a ``FeedForward``
    before using a Conditional Random Field model to predict a STREUSLE lextag for
    each token in the sequence. Decoding is constrained with the "BbIiOo_~"
    tagging scheme.

    This is mostly the same as the StreusleTagger, except that the STREUSLE 2.x dataset
    does not have lexcats. As a result, this model has no lexcat constraints. Also, the tagset
    is slightly different---instead of I_, I~, i_, i~, STREUSLE 2.x uses Ī, Ĩ, ī, ĩ.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to constrain the CRF decoding.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    dropout:  ``float``, optional (default=``None``)
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder = None,
                 label_namespace: str = "labels",
                 feedforward: Optional[FeedForward] = None,
                 include_start_end_transitions: bool = True,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        self.encoder = encoder
        if self.encoder is not None:
            encoder_output_dim = self.encoder.get_output_dim()
        else:
            encoder_output_dim = self.text_field_embedder.get_output_dim()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = encoder_output_dim
        self.tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                           self.num_tags))
        self._label_namespace = label_namespace
        labels = self.vocab.get_index_to_token_vocabulary(self._label_namespace)
        constraints = streusle_allowed_transitions(labels)

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions)

        self.accuracy_metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        if encoder is not None:
            check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                                   "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold lextags of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the example.

        Returns
        -------
        An output dictionary consisting of:

        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        if self.encoder:
            encoded_text = self.encoder(embedded_text_input, mask)
        else:
            encoded_text = embedded_text_input

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)

        best_paths = self.crf.viterbi_tags(logits, mask)
        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {
                "mask": mask,
                "tags": predicted_tags,
                "tokens": [instance_metadata["tokens"] for instance_metadata in metadata]
        }

        if tags is not None:
            # Add gold tags if they exist
            output["gold_tags"] = tags

            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.accuracy_metrics.values():
                metric(class_probabilities, tags, mask.float())
        return output

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` and ``output_dict["gold_tags"]`` are lists of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        mask = output_dict.pop("mask")
        lengths = util.get_lengths_from_binary_sequence_mask(mask)
        for key in "tags", "gold_tags":
            tags = output_dict.pop(key, None)
            if tags is not None:
                # TODO (nfliu): figure out why this is sometimes a tensor and sometimes a list.
                if torch.is_tensor(tags):
                    tags = tags.cpu().detach().numpy()
                output_dict[key] = [
                        [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                         for tag in instance_tags[:length]]
                        for instance_tags, length in zip(tags, lengths)]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.accuracy_metrics.items()}
        return metrics_to_return

def streusle_allowed_transitions(labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels, returns the allowed transitions when tagging STREUSLE lextags. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
        else:
            from_tag = from_label.split("-")[0]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
            else:
                to_tag = to_label.split("-")[0]
            if is_streusle_transition_allowed(from_tag, to_tag):
                allowed.append((from_label_index, to_label_index))
    return allowed

def is_streusle_transition_allowed(from_tag: str,
                                   to_tag: str):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the STREUSLE tagging scheme.

    Parameters
    ----------
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``Ĩ-V-v.cognition``, the ``from_tag`` is ``Ĩ``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``Ĩ-V-v.cognition``, the ``to_tag`` is ``Ĩ``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if from_tag not in ('O', 'B', 'Ī', 'Ĩ', 'o', 'b', 'ī', 'ĩ', 'START', 'END'):
        raise ValueError("Got invalid from_tag {}".format(from_tag))
    if to_tag not in ('O', 'B', 'Ī', 'Ĩ', 'o', 'b', 'ī', 'ĩ', 'START', 'END'):
        raise ValueError("Got invalid to_tag {}".format(to_tag))

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if from_tag == "START":
        return to_tag in ('O', 'B')
    if to_tag == "END":
        return from_tag in ('O', 'Ī', 'Ĩ')
    return any([
            # B can transition to o-*, b-*, Ī-*, or Ĩ-*
            # o can transition to o-*, b-*, Ī-*, or Ĩ-*
            from_tag in ('B', 'o') and to_tag in ('o', 'b', 'Ī', 'Ĩ'),
            # b can transition to ī or ĩ, but only if the entity tags match
            from_tag in ('b',) and to_tag in ('ī', 'ĩ'),
            # O can transition to O-*, B-*, or END
            from_tag in ('O',) and to_tag in ('O', 'B'),
            # Ī, Ĩ can transition to all tags except ī-* or ĩ-*
            from_tag in ('Ī', 'Ĩ') and to_tag not in ('ī', 'ĩ'),
            # ī, ĩ can transition to all tags except O, B
            from_tag in ('ī', 'ĩ') and to_tag not in ('O', 'B'),
    ])
