# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list

from streusle_tagger.dataset_readers import Streusle2DatasetReader


class TestStreusle2DatasetReader():
    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file(self, lazy):
        reader = Streusle2DatasetReader(lazy=lazy)
        instances = ensure_list(reader.read(str('fixtures/data/streusle2.tags')))
        assert len(instances) == 6

        instance = instances[0]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                'Highly', 'recommended']
        assert fields["tags"].labels == ["O", "O-communication"]

        instance = instances[1]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "My", "8", "year", "old", "daughter", "loves", "this", "place", "."]
        assert fields["tags"].labels == ["O", "O", "B-PERSON", "Ī", "O-PERSON",
                                         "O-emotion", "O", "O-LOCATION", "O"]

        instance = instances[2]
        fields = instance.fields
        assert [token.text for token in fields["tokens"]] == [
                "The", "best", "climbing", "club", "around", "."]
        assert fields["tags"].labels == ["O", "O", "O-ACT", "O-GROUP", "O", "O"]
