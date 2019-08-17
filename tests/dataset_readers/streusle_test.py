# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list

from streusle_tagger.dataset_readers import StreusleDatasetReader


class TestStreusleDatasetReader():
    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file(self, lazy):
        reader = StreusleDatasetReader(lazy=lazy)
        assert reader._upos_predictor is None
        instances = ensure_list(reader.read(str('fixtures/data/streusle.json')))
        assert reader._upos_predictor is None
        assert len(instances) == 3

        instance = instances[0]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                'Have', 'a', 'real', 'mechanic', 'check', 'before', 'you', 'buy', '!!!!']
        assert fields["metadata"]["upos_tags"] == [
                'VERB', 'DET', 'ADJ', 'NOUN', 'VERB', 'SCONJ', 'PRON', 'VERB', 'PUNCT']
        assert fields["tags"].labels == ["B-V-v.social", "o-DET", "o-ADJ",
                                         "o-N-n.PERSON", "I~-V-v.cognition",
                                         "O-P-p.Time", "O-PRON", "O-V-v.possession",
                                         "O-PUNCT"]

        instance = instances[1]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "Very", "good", "with", "my", "5", "year", "old", "daughter", "."]
        assert fields["metadata"]["upos_tags"] == [
                'ADV', 'ADJ', 'ADP', 'PRON', 'NUM', 'NOUN', 'ADJ', 'NOUN', 'PUNCT']
        assert fields["tags"].labels == ["O-ADV", "O-ADJ", "O-P-??",
                                         "O-PRON.POSS-p.SocialRel|p.Gestalt", "O-NUM",
                                         "B-N-n.PERSON", "I_", "O-N-n.PERSON", "O-PUNCT"]

        instance = instances[2]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                'After', 'firing', 'this', 'company', 'my', 'next', 'pool',
                'service', 'found', 'the', 'filters', 'had', 'not', 'been',
                'cleaned', 'as', 'they', 'should', 'have', 'been', '.']
        assert fields["metadata"]["upos_tags"] == [
                'SCONJ', 'VERB', 'DET', 'NOUN', 'PRON', 'ADJ', 'NOUN', 'NOUN', 'VERB',
                'DET', 'NOUN', 'AUX', 'PART', 'AUX', 'VERB', 'SCONJ', 'PRON', 'AUX',
                'AUX', 'AUX', 'PUNCT']
        assert fields["tags"].labels == ["O-P-p.Time", "O-V-v.social", "O-DET",
                                         "O-N-n.GROUP", "O-PRON.POSS-p.OrgRole|p.Gestalt",
                                         "O-ADJ", "B-N-n.ACT", "I_", "O-V-v.cognition",
                                         "O-DET", "O-N-n.ARTIFACT", "O-AUX", "O-ADV",
                                         "O-AUX", "O-V-v.change",
                                         "O-P-p.ComparisonRef", "O-PRON", "O-AUX",
                                         "O-AUX", "O-AUX", "O-PUNCT"]

    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file_predicted_upos(self, lazy):
        reader = StreusleDatasetReader(use_predicted_upos=True,
                                       lazy=lazy)
        assert reader._upos_predictor is None
        instances = ensure_list(reader.read(str('fixtures/data/streusle.json')))
        assert reader._upos_predictor is not None
        assert len(instances) == 3

        instance = instances[0]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                'Have', 'a', 'real', 'mechanic', 'check', 'before', 'you', 'buy', '!!!!']
        assert fields["metadata"]["upos_tags"] == [
                'VERB', 'DET', 'ADJ', 'ADJ', 'NOUN', 'SCONJ', 'PRON', 'VERB', 'PUNCT']
        assert fields["tags"].labels == ["B-V-v.social", "o-DET", "o-ADJ",
                                         "o-N-n.PERSON", "I~-V-v.cognition",
                                         "O-P-p.Time", "O-PRON", "O-V-v.possession",
                                         "O-PUNCT"]

        instance = instances[1]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "Very", "good", "with", "my", "5", "year", "old", "daughter", "."]
        assert fields["metadata"]["upos_tags"] == [
                'ADV', 'ADJ', 'ADP', 'PRON', 'NUM', 'NOUN', 'ADJ', 'NOUN', 'PUNCT']
        assert fields["tags"].labels == ["O-ADV", "O-ADJ", "O-P-??",
                                         "O-PRON.POSS-p.SocialRel|p.Gestalt", "O-NUM",
                                         "B-N-n.PERSON", "I_", "O-N-n.PERSON", "O-PUNCT"]

        instance = instances[2]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                'After', 'firing', 'this', 'company', 'my', 'next', 'pool',
                'service', 'found', 'the', 'filters', 'had', 'not', 'been',
                'cleaned', 'as', 'they', 'should', 'have', 'been', '.']
        assert fields["metadata"]["upos_tags"] == [
            'SCONJ', 'VERB', 'DET', 'NOUN', 'PRON', 'ADJ', 'NOUN', 'NOUN', 'VERB',
            'DET', 'NOUN', 'AUX', 'PART', 'AUX', 'VERB', 'SCONJ', 'PRON', 'AUX',
            'AUX', 'AUX', 'PUNCT']
        assert fields["tags"].labels == ["O-P-p.Time", "O-V-v.social", "O-DET",
                                         "O-N-n.GROUP", "O-PRON.POSS-p.OrgRole|p.Gestalt",
                                         "O-ADJ", "B-N-n.ACT", "I_", "O-V-v.cognition",
                                         "O-DET", "O-N-n.ARTIFACT", "O-AUX", "O-ADV",
                                         "O-AUX", "O-V-v.change",
                                         "O-P-p.ComparisonRef", "O-PRON", "O-AUX",
                                         "O-AUX", "O-AUX", "O-PUNCT"]
