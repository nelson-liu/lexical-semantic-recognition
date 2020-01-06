# pylint: disable=invalid-name,protected-access
from flaky import flaky

from allennlp.common.testing import ModelTestCase


class Streusle2TaggerTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('fixtures/streusle2_tagger/experiment.json',
                          'fixtures/data/streusle2.tags')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 6
        assert len(tags[0]) == 2
        assert len(tags[1]) == 9
        assert len(tags[2]) == 6
