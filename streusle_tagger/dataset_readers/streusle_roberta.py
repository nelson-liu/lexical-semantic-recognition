from typing import Dict, List
from copy import deepcopy
import json
import logging

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
import numpy as np
from overrides import overrides
import stanfordnlp
from transformers import AutoTokenizer

from streusle_tagger.data import SequentialArrayField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("streusle_roberta")
class StreusleRobertaDatasetReader(DatasetReader):
    """
    Reads data from the STREUSLE dataset and produces instances with tokens and
    their associated lextags. This reader produces tokens that are suitable for
    encoding with RoBERTa.

    Parameters
    ----------
    roberta_type: ``str``
        The type of RoBERTa model to use (``base`` or ``large``).
    max_seq_length: ``int``, optional (default = ``512``)
        The maximum number of wordpieces after tokenization. Shorter sequences
        will be padded to this value.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    use_predicted_upos: ``bool``, optional (default=``False``)
        Use predicted UPOS tags from StanfordNLP instead of the gold
        UPOS tags in the STREUSLE data.
    use_predicted_lemmas: ``bool``, optional (default=``False``)
        Use predicted lemmas from StanfordNLP instead of the gold
        lemmas in the STREUSLE data.
    """
    def __init__(self,
                 roberta_type: str,
                 max_seq_length: int = 512,
                 label_namespace: str = "labels",
                 use_predicted_upos: bool = False,
                 use_predicted_lemmas: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer = AutoTokenizer.from_pretrained(f"roberta-{roberta_type}")
        self.max_seq_length = max_seq_length

        self._use_predicted_upos = use_predicted_upos
        self._use_predicted_lemmas = use_predicted_lemmas
        # We initialize this in text_to_instance, if necessary.
        self._upos_predictor = None
        self._lemma_predictor = None
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)

        with open(file_path, 'r') as tagging_file:
            tagging_data = json.load(tagging_file)
            for instance in tagging_data:
                # Get the tokens
                tokens = [x["word"] for x in instance["toks"]]
                # Get their associated upos
                upos_tags = [x["upos"] for x in instance["toks"]]

                # Get their associated lemma
                lemmas = [x["lemma"] for x in instance["toks"]]
                # Get their associated lextag
                labels = [x["lextag"] for x in instance["toks"]]
                yield self.text_to_instance(tokens=tokens,
                                            upos_tags=upos_tags,
                                            lemmas=lemmas,
                                            streusle_lextags=labels)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         upos_tags: List[str] = None,
                         lemmas: List[str] = None,
                         streusle_lextags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        upos_tags : ``List[str]``, optional, (default = None).
            The upos_tags for the tokens in a given sentence. If None,
            we use StanfordNLP to predict them. If self._use_predicted_upos,
            we use StanfordNLP to predict them (ignoring any provided here).
        lemmas : ``List[str]``, optional, (default = None).
            The lemmas for the tokens in a given sentence. If None,
            we use StanfordNLP to predict them. If self._use_predicted_lemmas,
            we use StanfordNLP to predict them (ignoring any provided here).
        streusle_lextags : ``List[str]``, optional, (default = None).
            The STREUSLE lextags associated with a token.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            tags : ``SequenceLabelField``
                The tags corresponding to the ``tag_label`` constructor argument.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        roberta_inputs = convert_tokens_to_roberta_inputs(tokens=tokens,
                                                          tokenizer=self.tokenizer,
                                                          max_seq_length=self.max_seq_length)
        metadata = {
                "tokens": tokens,
                "token_indices_to_wordpiece_indices": roberta_inputs["token_indices_to_wordpiece_indices"]
        }
        fields["token_indices_to_wordpiece_indices"] = SequentialArrayField(
                np.array(roberta_inputs["token_indices_to_wordpiece_indices"], dtype="int64"),
                "int64", padding_value=-1)
        fields["input_ids"] = SequentialArrayField(np.array(roberta_inputs["input_ids"], dtype="int64"), "int64")
        fields["input_mask"] = SequentialArrayField(np.array(roberta_inputs["input_mask"], dtype="int64"), "int64")

        if self._use_predicted_upos or upos_tags is None:
            if self._upos_predictor is None:
                # Initialize UPOS predictor.
                self._upos_predictor = stanfordnlp.Pipeline(processors="tokenize,pos",
                                                            tokenize_pretokenized=True)
            doc = self._upos_predictor([tokens])
            upos_tags = [word.upos for sent in doc.sentences for word in sent.words]
        # Check number of UPOS tags equals number of tokens.
        assert len(upos_tags) == len(tokens)
        metadata["upos_tags"] = upos_tags

        if self._use_predicted_lemmas or lemmas is None:
            if self._lemma_predictor is None:
                # Initialize LEMMAS predictor.
                self._lemma_predictor = stanfordnlp.Pipeline(processors="tokenize,lemma",
                                                             tokenize_pretokenized=True)
            doc = self._lemma_predictor([tokens])
            lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        # Check number of LEMMAS tags equals number of tokens.
        assert len(lemmas) == len(tokens)
        metadata["lemmas"] = lemmas

        fields["metadata"] = MetadataField(metadata)
        # Add "tag label" to instance
        if streusle_lextags is not None:
            fields['tags'] = SequenceLabelField(streusle_lextags,
                                                fields["token_indices_to_wordpiece_indices"],
                                                self.label_namespace)
        return Instance(fields)

def convert_tokens_to_roberta_inputs(tokens,
                                     tokenizer,
                                     max_seq_length=512,
                                     cls_token=None,
                                     sep_token=None,
                                     pad_token=None):
    tokens = deepcopy(tokens)
    if cls_token is None:
        cls_token = tokenizer.cls_token
    if sep_token is None:
        sep_token = tokenizer.sep_token
    if pad_token is None:
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    token_indices_to_wordpiece_indices = []
    wordpiece_tokens = []
    for word in tokens:
        token_indices_to_wordpiece_indices.append(len(wordpiece_tokens))
        word_tokens = tokenizer.tokenize(word)
        wordpiece_tokens.extend(word_tokens)
    assert len(token_indices_to_wordpiece_indices) == len(tokens)

    # RoBERTa has 3 special tokens (2 SEP, 1 CLS)
    special_tokens_count = 3
    if len(wordpiece_tokens) > max_seq_length - special_tokens_count:
        # Don't truncate, raise an error instead
        # tokens = tokens[:(max_seq_length - special_tokens_count)]
        raise ValueError(f"tokens is of length {len(tokens)}, but "
                         f"max sequence length is {max_seq_length}.")
    # RoBERTa uses an extra separator b/w pairs of sentences
    wordpiece_tokens += [sep_token]
    wordpiece_tokens += [sep_token]

    # Prepend the CLS token.
    wordpiece_tokens = [cls_token] + wordpiece_tokens

    input_ids = tokenizer.convert_tokens_to_ids(wordpiece_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return {
            "token_indices_to_wordpiece_indices": token_indices_to_wordpiece_indices,
            "input_ids": input_ids,
            "input_mask": input_mask
    }
