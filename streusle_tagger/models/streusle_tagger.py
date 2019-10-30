from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import softmax

ALL_UPOS = {'X', 'INTJ', 'VERB', 'ADV', 'CCONJ', 'PUNCT', 'ADP',
            'NOUN', 'SYM', 'ADJ', 'PROPN', 'DET', 'PART', 'PRON', 'SCONJ', 'NUM', 'AUX'}
ALL_LEXCATS = {'N', 'INTJ', 'INF.P', 'V', 'AUX', 'PP', 'PUNCT',
               'POSS', 'X', 'PRON.POSS', 'SYM', 'PRON', 'SCONJ',
               'NUM', 'DISC', 'ADV', 'CCONJ', 'P', 'ADJ', 'DET', 'INF'}
SPECIAL_LEMMAS = {"to", "be", "versus"}
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("streusle_tagger")
class StreusleTagger(Model):
    """
    The ``StreusleTagger`` embeds a sequence of text before (optionally)
    encoding it with a ``Seq2SeqEncoder`` and passing it through a ``FeedForward``
    before using a Conditional Random Field model to predict a STREUSLE lextag for
    each token in the sequence. Decoding is constrained with the "BbIiOo_~"
    tagging scheme.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        The encoder that we will use in between embedding tokens and predicting output tags.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    dropout:  ``float``, optional (default=``None``)
    use_upos_constraints : ``bool``, optional (default=``True``)
        Whether to use UPOS constraints. If True, model shoudl recieve UPOS as input.
    use_lemma_constraints : ``bool``, optional (default=``True``)
        Whether to use lemma constraints. If True, model shoudl recieve lemmas as input.
        If this is true, then use_upos_constraints must be true as well.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.

    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder = None,
                 feedforward: Optional[FeedForward] = None,
                 include_start_end_transitions: bool = True,
                 dropout: Optional[float] = None,
                 use_upos_constraints: bool = True,
                 use_lemma_constraints: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_mwe_lexcat_tags = self.vocab.get_vocab_size("mwe_lexcat_tags")
        self.num_ss_tags = self.vocab.get_vocab_size("ss_tags")
        self.num_ss2_tags = self.vocab.get_vocab_size("ss2_tags")

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
        self.mwe_lexcat_tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                                      self.num_mwe_lexcat_tags))
        self.ss_tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                              self.num_ss_tags))
        self.ss2_tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                               self.num_ss2_tags))
        mwe_lexcat_tags = self.vocab.get_index_to_token_vocabulary("mwe_lexcat_tags")
        constraints = streusle_allowed_transitions(mwe_lexcat_tags)

        self.use_upos_constraints = use_upos_constraints
        self.use_lemma_constraints = use_lemma_constraints

        if self.use_lemma_constraints and not self.use_upos_constraints:
            raise ConfigurationError("If lemma constraints are applied, UPOS constraints must be applied as well.")

        if self.use_upos_constraints:
            # Get a dict with a mapping from UPOS to allowed LEXCAT here.
            self._upos_to_allowed_lexcats: Dict[str, Set[str]] = get_upos_allowed_lexcats(
                    stronger_constraints=self.use_lemma_constraints)
            # Dict with a amapping from UPOS to dictionary of [UPOS, list of additionally allowed LEXCATS]
            self._lemma_to_allowed_lexcats: Dict[str, Dict[str, List[str]]] = get_lemma_allowed_lexcats()

            # Use labels and the upos_to_allowed_lexcats to get a dict with
            # a mapping from UPOS to a mask with 1 at allowed label indices and 0 at
            # disallowed label indices.
            self._upos_to_label_mask: Dict[str, torch.Tensor] = {}
            for upos in ALL_UPOS:
                # Shape: (num_labels,)
                upos_label_mask = torch.zeros(
                        len(mwe_lexcat_tags),
                        device=next(self.mwe_lexcat_tag_projection_layer.parameters()).device)
                # Go through the labels and indices and fill in the values that are allowed.
                for label_index, label in mwe_lexcat_tags.items():
                    if len(label.split("-")) == 1:
                        upos_label_mask[label_index] = 1
                        continue
                    label_lexcat = label.split("-")[1]
                    if not label.startswith("O-") and not label.startswith("o-"):
                        # Label does not start with O-/o-, always allowed.
                        upos_label_mask[label_index] = 1
                    elif label_lexcat in self._upos_to_allowed_lexcats[upos]:
                        # Label starts with O-/o-, but the lexcat is in allowed
                        # lexcats for the current upos.
                        upos_label_mask[label_index] = 1
                self._upos_to_label_mask[upos] = upos_label_mask

            # Use labels and the lemma_to_allowed_lexcats to get a dict with
            # a mapping from lemma to a mask with 1 at an _additionally_ allowed label index
            # and 0 at disallowed label indices. If lemma_to_label_mask has a 0, and upos_to_label_mask
            # has a 0, the lexcat is not allowed for the (upos, lemma). If either lemma_to_label_mask or
            # upos_to_label_mask has a 1, the lexcat is allowed for the (upos, lemma) pair.
            self._lemma_upos_to_label_mask: Dict[Tuple[str, str], torch.Tensor] = {}
            for lemma in SPECIAL_LEMMAS:
                for upos_tag in ALL_UPOS:
                    # No additional constraints, should be all zero
                    if upos_tag not in self._lemma_to_allowed_lexcats[lemma]:
                        continue
                    # Shape: (num_labels,)
                    lemma_upos_label_mask = torch.zeros(
                            len(mwe_lexcat_tags),
                            device=next(self.mwe_lexcat_tag_projection_layer.parameters()).device)
                    # Go through the labels and indices and fill in the values that are allowed.
                    for label_index, label in mwe_lexcat_tags.items():
                        # For ~i, etc. tags. We don't deal with them here.
                        if len(label.split("-")) == 1:
                            continue
                        label_lexcat = label.split("-")[1]
                        if not label.startswith("O-") and not label.startswith("o-"):
                            # Label does not start with O-/o-, so we don't deal with it here
                            continue
                        if label_lexcat in self._lemma_to_allowed_lexcats[lemma][upos_tag]:
                            # Label starts with O-/o-, but the lexcat is in allowed
                            # lexcats for the current upos.
                            lemma_upos_label_mask[label_index] = 1
                    self._lemma_upos_to_label_mask[(lemma, upos_tag)] = lemma_upos_label_mask

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_mwe_lexcat_tags, constraints,
                include_start_end_transitions=include_start_end_transitions)

        self.metrics = {
                "mwe_lexcat_accuracy": CategoricalAccuracy(),
                "mwe_lexcat_accuracy3": CategoricalAccuracy(top_k=3),
                "ss_accuracy": CategoricalAccuracy(),
                "ss_accuracy3": CategoricalAccuracy(top_k=3),
                "ss2_accuracy": CategoricalAccuracy(),
                "ss2_accuracy3": CategoricalAccuracy(top_k=3),
                "combined_em_accuracy": BooleanAccuracy(),
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
                mwe_lexcat_tags: torch.LongTensor = None,
                ss_tags: torch.LongTensor = None,
                ss2_tags: torch.LongTensor = None,
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
        mwe_lexcat_tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold mwe-lexcat tags of shape
            ``(batch_size, num_tokens)``.
        ss_tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold ss tags of shape
            ``(batch_size, num_tokens)``.
        ss2_tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold ss2 tags of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the example.

        Returns
        -------
        An output dictionary consisting of:

        constrained_logits : ``torch.FloatTensor``
            The constrained logits that are the output of the ``tag_projection_layer``
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

        mwe_lexcat_logits = self.mwe_lexcat_tag_projection_layer(encoded_text)
        ss_logits = self.ss_tag_projection_layer(encoded_text)
        ss2_logits = self.ss2_tag_projection_layer(encoded_text)

        # initial mask is unmasked
        batch_upos_constraint_mask = torch.ones_like(mwe_lexcat_logits)
        if self.use_upos_constraints:
            # List of length (batch_size,), where each inner list is a list of
            # the UPOS tags for the associated token sequence.
            batch_upos_tags = [instance_metadata["upos_tags"] for instance_metadata in metadata]

            # List of length (batch_size,), where each inner list is a list of
            # the lemmas for the associated token sequence.
            if self.use_lemma_constraints:
                batch_lemmas = [instance_metadata["lemmas"] for
                                instance_metadata in metadata]
            else:
                batch_lemmas = [([None] * len(instance_metadata["upos_tags"])) for
                                instance_metadata in metadata]

            # Get a (batch_size, max_sequence_length, num_tags) mask with "1" in
            # tags that are allowed for a given UPOS, and "0" for tags that are
            # disallowed for an even UPOS.
            batch_upos_constraint_mask = self.get_upos_constraint_mask(batch_upos_tags=batch_upos_tags,
                                                                       batch_lemmas=batch_lemmas)

        constrained_mwe_lexcat_logits = util.replace_masked_values(mwe_lexcat_logits,
                                                                   batch_upos_constraint_mask,
                                                                   -1e8)

        best_paths = self.crf.viterbi_tags(mwe_lexcat_logits, mask)
        # Just get the tags and ignore the score.
        predicted_mwe_lexcat_tags = [x for x, y in best_paths]

        output = {
                "mask": mask,
                "mwe_lexcat_tags": predicted_mwe_lexcat_tags,
                "tokens": [instance_metadata["tokens"] for instance_metadata in metadata]
        }

        if self.use_upos_constraints:
            output["constrained_mwe_lexcat_logits"] = constrained_mwe_lexcat_logits
            output["upos_tags"] = batch_upos_tags

        ss_class_probabilities = softmax(ss_logits, dim=-1)
        output["ss_class_probabilities"] = ss_class_probabilities
        ss2_class_probabilities = softmax(ss2_logits, dim=-1)
        output["ss2_class_probabilities"] = ss2_class_probabilities

        if mwe_lexcat_tags is not None and ss_tags is not None and ss2_tags is not None:
            # Add negative log-likelihood of mwe lexcat tags as loss
            mwe_lexcat_nll = -self.crf(constrained_mwe_lexcat_logits, mwe_lexcat_tags, mask)

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            mwe_lexcat_class_probabilities = constrained_mwe_lexcat_logits * 0.
            for i, instance_tags in enumerate(predicted_mwe_lexcat_tags):
                for j, tag_id in enumerate(instance_tags):
                    mwe_lexcat_class_probabilities[i, j, tag_id] = 1
            output["mwe_lexcat_class_probabilities"] = mwe_lexcat_class_probabilities
            self.metrics["mwe_lexcat_accuracy"](mwe_lexcat_class_probabilities, mwe_lexcat_tags, mask.float())
            self.metrics["mwe_lexcat_accuracy3"](mwe_lexcat_class_probabilities, mwe_lexcat_tags, mask.float())

            # Add NLL of the ss tags as loss
            ss_nll = util.sequence_cross_entropy_with_logits(ss_logits, ss_tags, mask)
            self.metrics["ss_accuracy"](ss_class_probabilities, ss_tags, mask.float())
            self.metrics["ss_accuracy3"](ss_class_probabilities, ss_tags, mask.float())

            # Add NLL of the ss2 tags as loss
            ss2_nll = util.sequence_cross_entropy_with_logits(ss2_logits, ss2_tags, mask)
            self.metrics["ss2_accuracy"](ss2_class_probabilities, ss2_tags, mask.float())
            self.metrics["ss2_accuracy3"](ss2_class_probabilities, ss2_tags, mask.float())

            # Shape: (batch_size, sequence_length, 3)
            # 3 because there is one number of mwe_lexcat label, one for ss, and one for ss2
            combined_predictions = torch.stack([mwe_lexcat_logits.argmax(-1),
                                                ss_logits.argmax(-1),
                                                ss2_logits.argmax(-1)], dim=-1)
            combined_gold = torch.stack([mwe_lexcat_tags, ss_tags, ss2_tags], dim=-1)
            for instance_mask, instance_predictions, instance_gold in zip(mask,
                                                                          combined_predictions,
                                                                          combined_gold):
                for token_mask, token_prediction, token_gold in zip(instance_mask,
                                                                    instance_predictions,
                                                                    instance_gold):
                    # Only calculate accuracy on unmasked items.
                    if token_mask.item() == 1:
                        self.metrics["combined_em_accuracy"](token_prediction, token_gold)

            # Aggregate losses
            output["loss"] = mwe_lexcat_nll + ss_nll + ss2_nll
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["mwe_lexcat_tags"] = [
                [self.vocab.get_token_from_index(mwe_lexcat_tag, namespace="mwe_lexcat_tags")
                 for mwe_lexcat_tag in instance_mwe_lexcat_tags]
                for instance_mwe_lexcat_tags in output_dict["mwe_lexcat_tags"]]

        mask = output_dict["mask"].cpu().data.numpy()
        all_ss_predictions = output_dict["ss_class_probabilities"]
        all_ss_predictions = all_ss_predictions.cpu().data.numpy()
        if all_ss_predictions.ndim == 3:
            ss_predictions_list = [all_ss_predictions[i] for i in range(all_ss_predictions.shape[0])]
            mask_list = [mask[i] for i in range(mask.shape[0])]
        else:
            ss_predictions_list = [all_ss_predictions]
            mask_list = [mask]
        all_ss_tags = []
        for mask, ss_predictions in zip(mask_list, ss_predictions_list):
            ss_argmax_indices = numpy.argmax(ss_predictions, axis=-1)
            ss_tags = [self.vocab.get_token_from_index(x, namespace="ss_tags") for x in ss_argmax_indices]
            # Remove masked items
            ss_tags = ss_tags[:mask.sum()]
            all_ss_tags.append(ss_tags)
        output_dict["ss_tags"] = all_ss_tags

        all_ss2_predictions = output_dict["ss2_class_probabilities"]
        all_ss2_predictions = all_ss2_predictions.cpu().data.numpy()
        if all_ss2_predictions.ndim == 3:
            ss2_predictions_list = [all_ss2_predictions[i] for i in range(all_ss2_predictions.shape[0])]
        else:
            ss2_predictions_list = [all_ss2_predictions]
        all_ss2_tags = []
        for mask, ss2_predictions in zip(mask_list, ss2_predictions_list):
            ss2_argmax_indices = numpy.argmax(ss2_predictions, axis=-1)
            ss2_tags = [self.vocab.get_token_from_index(x, namespace="ss2_tags") for x in ss2_argmax_indices]
            # Remove masked items
            ss2_tags = ss2_tags[:mask.sum()]
            all_ss2_tags.append(ss2_tags)
        output_dict["ss2_tags"] = all_ss2_tags
        # Construct the lextags from the predicted "mwe_lexcat_tags", "ss2_tags", and "ss_tags"
        all_lextags = []
        assert len(output_dict["mwe_lexcat_tags"]) == len(output_dict["ss_tags"])
        assert len(output_dict["ss_tags"]) == len(output_dict["ss2_tags"])
        for mwe_lexcat_tags, ss_tags, ss2_tags in zip(output_dict["mwe_lexcat_tags"],
                                                      output_dict["ss_tags"],
                                                      output_dict["ss2_tags"]):
            lextags = []
            assert len(mwe_lexcat_tags) == len(ss_tags)
            assert len(ss_tags) == len(ss2_tags)
            for mwe_lexcat_tag, ss_tag, ss2_tag in zip(mwe_lexcat_tags, ss_tags, ss2_tags):
                lextag = mwe_lexcat_tag
                if ss_tag != "@@<NO_SS>@@":
                    lextag = f"{lextag}-{ss_tag}"
                    if ss2_tag != "@@<NO_SS2>@@":
                        lextag = f"{lextag}|{ss2_tag}"
                lextags.append(lextag)
            all_lextags.append(lextags)
        output_dict["tags"] = all_lextags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return

    def get_upos_constraint_mask(self,
                                 batch_upos_tags: List[List[str]],
                                 batch_lemmas: List[List[str]]):
        """
        Given POS tags and lemmas for a batch, return a mask of shape
        (batch_size, max_sequence_length, num_tags) mask with "1" in
        tags that are allowed for a given UPOS, and "0" for tags that are
        disallowed for a given UPOS.

        Parameters
        ----------
        batch_upos_tags: ``List[List[str]]``, required
            UPOS tags for a batch.
        batch_lemmas: ``List[List[str]]``, required
            Lemmas for a batch.

        Returns
        -------
        ``Tensor``, shape (batch_size, max_sequence_length, num_tags)
            A mask over the logits, with 1 in positions where a tag is allowed
            for its UPOS and 0 in positions where a tag is allowed for its UPOS.
        """
        # TODO(nfliu): this is pretty inefficient, maybe there's someway to make it batched?
        # Shape: (batch_size, max_sequence_length, num_tags)
        upos_constraint_mask = torch.ones(
                len(batch_upos_tags),
                len(max(batch_upos_tags, key=len)),
                self.num_mwe_lexcat_tags,
                device=next(self.mwe_lexcat_tag_projection_layer.parameters()).device) * -1e32
        # Iterate over the batch
        for example_index, (example_upos_tags, example_lemmas) in enumerate(
                zip(batch_upos_tags, batch_lemmas)):
            # Shape of example_constraint_mask: (max_sequence_length, num_tags)
            # Iterate over the upos tags for the example
            example_constraint_mask = upos_constraint_mask[example_index]
            for timestep_index, (timestep_upos_tag, timestep_lemma) in enumerate(  # pylint: disable=unused-variable
                    zip(example_upos_tags, example_lemmas)):
                # Shape of timestep_constraint_mask: (num_tags,)
                upos_constraint = self._upos_to_label_mask[timestep_upos_tag]
                lemma_constraint = self._lemma_upos_to_label_mask.get((timestep_lemma, timestep_upos_tag),
                                                                      torch.zeros_like(upos_constraint))
                example_constraint_mask[timestep_index] = (upos_constraint.long() |
                                                           lemma_constraint.long()).float()
        return upos_constraint_mask

def get_lemma_allowed_lexcats():
    lemmas_to_constraints = {}
    # (POS, LEXCAT)
    lemmas_to_constraints["for"] = {"SCONJ": {"INF", "INF.P"}}
    lemmas_to_constraints["to"] = {"PART": {"INF", "INF.P"}}
    lemmas_to_constraints["be"] = {"AUX": {"V"}}
    lemmas_to_constraints["versus"] = {"ADP": {"CCONJ"}, "SCONJ": {"CCONJ"}}
    logger.info("Additionally allowed lexcats for each UPOS and each lemma")
    json_lemmas_to_constraints = {}
    for lemma in lemmas_to_constraints:
        json_lemmas_to_constraints[lemma] = dict()
        for upos in lemmas_to_constraints[lemma]:
            json_lemmas_to_constraints[lemma][upos] = sorted(list(lemmas_to_constraints[lemma][upos]))
    logger.info(json.dumps(json_lemmas_to_constraints, indent=2))
    return lemmas_to_constraints

def get_upos_allowed_lexcats(stronger_constraints=False):
    """
    stronger_constraints: bool (optional, default=False)
        If True, return mark a LEXCAT as allowed for a particular UPOS
        only if the LEXCAT is _always_a llowed for that UPOS. For instance,
        if this is True, then LEXCAT "AUX" will not be marked as allowed for
        upos "V", since it's only ok when the lemma is "be". If the argument
        is false, then LEXCAT "AUX" will be marked as allowed for upos "V".
    """
    if stronger_constraints:
        logger.info("Using UPOS constraints that are stronger than necessary "
                    "(probably because we are also using lemma constraints).")
    # pylint: disable=too-many-return-statements
    def is_allowed(upos, lexcat, stronger):
        if lexcat.endswith('!@'):
            return True
        if upos == lexcat:
            return True
        if (upos, lexcat) in {('NOUN', 'N'), ('PROPN', 'N'), ('VERB', 'V'),
                              ('ADP', 'P'), ('ADV', 'P'), ('SCONJ', 'P'),
                              ('ADP', 'DISC'), ('ADV', 'DISC'), ('SCONJ', 'DISC'),
                              ('PART', 'POSS')}:
            return True
        mismatch_ok = False
        if lexcat.startswith('INF'):
            # LC INF/INF.P and UPOS SCONJ are only ok if the lemma is "for".
            if upos == "SCONJ" and stronger is False:
                mismatch_ok = True
            # LC INF/INF.P and UPOS PART are only ok if the lemma is "to".
            if upos == "PART" and stronger is False:
                mismatch_ok = True
        # LC V and UPOS AUX are ok only if the lemma is "be".
        if upos == "AUX" and lexcat == "V" and stronger is False:
            mismatch_ok = True
        if upos == 'PRON':
            if lexcat in ('PRON', 'PRON.POSS'):
                mismatch_ok = True
        if lexcat == 'ADV':
            if upos in ('ADV', 'PART'):
                mismatch_ok = True
        # LC CCONJ and UPOS ADP are ok only if the lemma is "versus"
        if upos == 'ADP' and lexcat == 'CCONJ' and stronger is False:
            mismatch_ok = True
        return mismatch_ok

    allowed_combinations = {}
    for lexcat in ALL_LEXCATS:
        for universal_pos in ALL_UPOS:
            if is_allowed(universal_pos, lexcat, stronger_constraints):
                if universal_pos not in allowed_combinations:
                    allowed_combinations[universal_pos] = set()
                allowed_combinations[universal_pos].add(lexcat)
    logger.info("Allowed lexcats for each UPOS:")
    json_allowed_combinations = {upos: sorted(list(lexcats)) for
                                 upos, lexcats in allowed_combinations.items()}
    logger.info(json.dumps(json_allowed_combinations, indent=2))
    return allowed_combinations


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
        label is ``I~-V-v.cognition``, the ``from_tag`` is ``I~``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I~-V-v.cognition``, the ``to_tag`` is ``I~``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if from_tag not in ('O', 'B', 'I_', 'I~', 'o', 'b', 'i_', 'i~', 'START', 'END'):
        raise ValueError("Got invalid from_tag {}".format(from_tag))
    if to_tag not in ('O', 'B', 'I_', 'I~', 'o', 'b', 'i_', 'i~', 'START', 'END'):
        raise ValueError("Got invalid to_tag {}".format(to_tag))

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if from_tag == "START":
        return to_tag in ('O', 'B')
    if to_tag == "END":
        return from_tag in ('O', 'I_', 'I~')
    return any([
            # B can transition to o-*, b-*, I_-*, or I~-*
            # o can transition to o-*, b-*, I_-*, or I~-*
            from_tag in ('B', 'o') and to_tag in ('o', 'b', 'I_', 'I~'),
            # b can transition to i_ or i~, but only if the entity tags match
            from_tag in ('b',) and to_tag in ('i_', 'i~'),
            # O can transition to O-*, B-*, or END
            from_tag in ('O',) and to_tag in ('O', 'B'),
            # I_, I~ can transition to all tags except i_-* or i~-*
            from_tag in ('I_', 'I~') and to_tag not in ('i_', 'i~'),
            # i_, i~ can transition to all tags except O, B
            from_tag in ('i_', 'i~') and to_tag not in ('O', 'B'),
    ])
