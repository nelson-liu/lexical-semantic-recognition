# pylint: disable=invalid-name,protected-access
from flaky import flaky

from allennlp.common.testing import ModelTestCase


class StreusleTaggerTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('fixtures/streusle_tagger/experiment_all_constraints.json',
                          'fixtures/data/streusle.json')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_simple_tagger_no_constraints_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            'fixtures/streusle_tagger/experiment_no_constraints.json')

    def test_simple_tagger_no_constraints_during_training_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            'fixtures/streusle_tagger/experiment_no_constraints_during_training.json')

    def test_simple_tagger_upos_constraints_only_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            'fixtures/streusle_tagger/experiment_upos_constraints_only.json')

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 3
        assert len(tags[0]) == 9
        assert len(tags[1]) == 9
        assert len(tags[2]) == 21
