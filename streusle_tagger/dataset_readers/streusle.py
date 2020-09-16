from typing import Dict, List
import json
import logging

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
import stanza

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("streusle")
class StreusleDatasetReader(DatasetReader):
    """
    Reads data from the STREUSLE dataset and produces instances with tokens and
    their associated lextags.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenized in the data file.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    use_predicted_upos: ``bool``, optional (default=``False``)
        Use predicted UPOS tags from Stanza instead of the gold
        UPOS tags in the STREUSLE data.
    use_predicted_lemmas: ``bool``, optional (default=``False``)
        Use predicted lemmas from Stanza instead of the gold
        lemmas in the STREUSLE data.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels",
                 use_predicted_upos: bool = False,
                 use_predicted_lemmas: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
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
            we use Stanza to predict them. If self._use_predicted_upos,
            we use Stanza to predict them (ignoring any provided here).
        lemmas : ``List[str]``, optional, (default = None).
            The lemmas for the tokens in a given sentence. If None,
            we use Stanza to predict them. If self._use_predicted_lemmas,
            we use Stanza to predict them (ignoring any provided here).
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
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        metadata = {"tokens": tokens}
        if self._use_predicted_upos or upos_tags is None:
            if self._upos_predictor is None:
                # Initialize UPOS predictor.
                self._upos_predictor = stanza.Pipeline(processors="tokenize,pos",
                                                            tokenize_pretokenized=True)
            doc = self._upos_predictor([tokens])
            upos_tags = [word.upos for sent in doc.sentences for word in sent.words]
        # Check number of UPOS tags equals number of tokens.
        assert len(upos_tags) == len(tokens)
        metadata["upos_tags"] = upos_tags

        if self._use_predicted_lemmas or lemmas is None:
            if self._lemma_predictor is None:
                # Initialize LEMMAS predictor.
                self._lemma_predictor = stanza.Pipeline(processors="tokenize,lemma",
                                                             tokenize_pretokenized=True)
            doc = self._lemma_predictor([tokens])
            lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        # Check number of LEMMAS tags equals number of tokens.
        assert len(lemmas) == len(tokens)
        metadata["lemmas"] = lemmas

        fields["metadata"] = MetadataField(metadata)
        # Add "tag label" to instance
        if streusle_lextags is not None:
            fields['tags'] = SequenceLabelField(streusle_lextags, text_field,
                                                self.label_namespace)
        return Instance(fields)
