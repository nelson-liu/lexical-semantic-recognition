# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestStreusleTaggerPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {"tokens": ["This", "is", "a", "sample", "sentence", "."]}

        archive = load_archive('fixtures/streusle_tagger/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'streusle-tagger')
        result = predictor.predict_json(inputs)
        tags_list = result.get("tags")
        for tag in tags_list:
            assert isinstance(tag, str)
            assert tag != ""

    def test_batch_prediction(self):
        inputs = [{"tokens": ["This", "is", "a", "sample", "sentence", "."]},
                  {"tokens": ["Another", "sample", "is", "provided", "."]}]

        archive = load_archive('fixtures/streusle_tagger/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'streusle-tagger')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2
