from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

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
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels",
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
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
                # Get their associated lextag
                labels = [x["lextag"] for x in instance["toks"]]
                yield self.text_to_instance(tokens,
                                            labels)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         streusle_lextags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
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

        # Add "tag label" to instance
        fields['tags'] = SequenceLabelField(streusle_lextags, text_field,
                                            self.label_namespace)
        return Instance(fields)
