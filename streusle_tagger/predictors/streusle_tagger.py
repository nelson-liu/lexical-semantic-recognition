from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('streusle-tagger')
class StreusleTaggerPredictor(Predictor):
    """"
    Predictor for the :class:`~allennlp.models.streusle_tagger.StreusleTagger` model.
    """
    def dump_line(self, outputs: JsonDict) -> str:
        if "mask" in outputs:
            return str(outputs["tags"][:sum(outputs["mask"])]) + "\n"
        else:
            return str(outputs["tags"]) + "\n"
