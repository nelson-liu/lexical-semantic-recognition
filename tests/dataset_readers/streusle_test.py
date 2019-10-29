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
        assert fields["mwe_lexcat_tags"].labels == ['B-V', 'o-DET', 'o-ADJ', 'o-N', 'I~-V', 'O-P',
                                                    'O-PRON', 'O-V', 'O-PUNCT']
        assert fields["ss_tags"].labels == ["v.social", "@@<NO_SS>@@", "@@<NO_SS>@@", "n.PERSON", "v.cognition",
                                            "p.Time", "@@<NO_SS>@@", "v.possession", "@@<NO_SS>@@"]
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@"]

        instance = instances[1]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "Very", "good", "with", "my", "5", "year", "old", "daughter", "."]
        assert fields["metadata"]["upos_tags"] == [
                'ADV', 'ADJ', 'ADP', 'PRON', 'NUM', 'NOUN', 'ADJ', 'NOUN', 'PUNCT']
        assert fields["mwe_lexcat_tags"].labels == ['O-ADV', 'O-ADJ', 'O-P', 'O-PRON.POSS', 'O-NUM', 'B-N',
                                                    'I_', 'O-N', 'O-PUNCT']
        assert fields["ss_tags"].labels == ["@@<NO_SS>@@", "@@<NO_SS>@@", "??", "p.SocialRel", "@@<NO_SS>@@",
                                            "n.PERSON", "@@<NO_SS>@@", "n.PERSON", "@@<NO_SS>@@"]
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "p.Gestalt",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@"]

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
        assert fields["mwe_lexcat_tags"].labels == ['O-P', 'O-V', 'O-DET', 'O-N', 'O-PRON.POSS', 'O-ADJ', 'B-N',
                                                    'I_', 'O-V', 'O-DET', 'O-N', 'O-AUX', 'O-ADV',
                                                    'O-AUX', 'O-V', 'O-P', 'O-PRON', 'O-AUX', 'O-AUX', 'O-AUX', 'O-PUNCT']
        assert fields["ss_tags"].labels == ['p.Time', 'v.social', '@@<NO_SS>@@', 'n.GROUP', 'p.OrgRole',
                                            '@@<NO_SS>@@', 'n.ACT', '@@<NO_SS>@@', 'v.cognition', '@@<NO_SS>@@',
                                            'n.ARTIFACT', '@@<NO_SS>@@', '@@<NO_SS>@@', '@@<NO_SS>@@', 'v.change',
                                            'p.ComparisonRef', '@@<NO_SS>@@', '@@<NO_SS>@@', '@@<NO_SS>@@',
                                            '@@<NO_SS>@@', '@@<NO_SS>@@']
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", '@@<NO_SS2>@@', "@@<NO_SS2>@@", 'p.Gestalt',
                                             '@@<NO_SS2>@@', "@@<NO_SS2>@@", '@@<NO_SS2>@@', "@@<NO_SS2>@@", '@@<NO_SS2>@@',
                                             "@@<NO_SS2>@@", '@@<NO_SS2>@@', '@@<NO_SS2>@@', '@@<NO_SS2>@@', "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", '@@<NO_SS2>@@', '@@<NO_SS2>@@', '@@<NO_SS2>@@',
                                             '@@<NO_SS2>@@', '@@<NO_SS2>@@']

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
        assert fields["mwe_lexcat_tags"].labels == ['B-V', 'o-DET', 'o-ADJ', 'o-N', 'I~-V', 'O-P',
                                                    'O-PRON', 'O-V', 'O-PUNCT']
        assert fields["ss_tags"].labels == ["v.social", "@@<NO_SS>@@", "@@<NO_SS>@@", "n.PERSON", "v.cognition",
                                            "p.Time", "@@<NO_SS>@@", "v.possession", "@@<NO_SS>@@"]
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@"]

        instance = instances[1]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "Very", "good", "with", "my", "5", "year", "old", "daughter", "."]
        assert fields["metadata"]["upos_tags"] == [
                'ADV', 'ADJ', 'ADP', 'PRON', 'NUM', 'NOUN', 'ADJ', 'NOUN', 'PUNCT']
        assert fields["mwe_lexcat_tags"].labels == ['O-ADV', 'O-ADJ', 'O-P', 'O-PRON.POSS', 'O-NUM', 'B-N',
                                                    'I_', 'O-N', 'O-PUNCT']
        assert fields["ss_tags"].labels == ["@@<NO_SS>@@", "@@<NO_SS>@@", "??", "p.SocialRel", "@@<NO_SS>@@",
                                            "n.PERSON", "@@<NO_SS>@@", "n.PERSON", "@@<NO_SS>@@"]
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "p.Gestalt",
                                             "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@", "@@<NO_SS2>@@"]

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
        assert fields["mwe_lexcat_tags"].labels == ['O-P', 'O-V', 'O-DET', 'O-N', 'O-PRON.POSS', 'O-ADJ', 'B-N',
                                                    'I_', 'O-V', 'O-DET', 'O-N', 'O-AUX', 'O-ADV',
                                                    'O-AUX', 'O-V', 'O-P', 'O-PRON', 'O-AUX', 'O-AUX', 'O-AUX', 'O-PUNCT']
        assert fields["ss_tags"].labels == ['p.Time', 'v.social', '@@<NO_SS>@@', 'n.GROUP', 'p.OrgRole',
                                            '@@<NO_SS>@@', 'n.ACT', '@@<NO_SS>@@', 'v.cognition', '@@<NO_SS>@@',
                                            'n.ARTIFACT', '@@<NO_SS>@@', '@@<NO_SS>@@', '@@<NO_SS>@@', 'v.change',
                                            'p.ComparisonRef', '@@<NO_SS>@@', '@@<NO_SS>@@', '@@<NO_SS>@@',
                                            '@@<NO_SS>@@', '@@<NO_SS>@@']
        assert fields["ss2_tags"].labels == ["@@<NO_SS2>@@", "@@<NO_SS2>@@", '@@<NO_SS2>@@', "@@<NO_SS2>@@", 'p.Gestalt',
                                             '@@<NO_SS2>@@', "@@<NO_SS2>@@", '@@<NO_SS2>@@', "@@<NO_SS2>@@", '@@<NO_SS2>@@',
                                             "@@<NO_SS2>@@", '@@<NO_SS2>@@', '@@<NO_SS2>@@', '@@<NO_SS2>@@', "@@<NO_SS2>@@",
                                             "@@<NO_SS2>@@", '@@<NO_SS2>@@', '@@<NO_SS2>@@', '@@<NO_SS2>@@',
                                             '@@<NO_SS2>@@', '@@<NO_SS2>@@']
