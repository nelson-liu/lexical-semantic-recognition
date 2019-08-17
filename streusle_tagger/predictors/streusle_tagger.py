from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('streusle-tagger')
class StreusleTaggerPredictor(Predictor):
    """"
    Predictor for the :class:`~allennlp.models.streusle_tagger.StreusleTagger` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"tokens": "[...]", "upos_tags": "[...]"}``.
        """
        tokens = json_dict["tokens"]
        upos_tags = json_dict.get("upos_tags", None)
        return self._dataset_reader.text_to_instance(tokens=tokens,
                                                     upos_tags=upos_tags)
