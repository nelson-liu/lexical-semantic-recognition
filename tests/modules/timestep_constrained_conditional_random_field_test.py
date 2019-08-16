# pylint: disable=no-self-use,invalid-name
import itertools

import torch

from allennlp.modules import ConditionalRandomField
from allennlp.common.testing import AllenNlpTestCase

from streusle_tagger.modules import TimestepConstrainedConditionalRandomField


class TestTimestepConstrainedConditionalRandomField(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.logits = torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ])
        self.tags = torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ])

        self.transitions = torch.Tensor([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = ConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        # Make a TimestepConstrainedConditionalRandomField with the same fixed
        # # transitions to compute the log_likelihood
        self.timestep_constrained_crf = TimestepConstrainedConditionalRandomField(5)
        self.timestep_constrained_crf.transitions = torch.nn.Parameter(self.transitions)
        self.timestep_constrained_crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.timestep_constrained_crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def test_viterbi_tags(self):
        mask = torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ])

        viterbi_path = self.crf.viterbi_tags(self.logits, mask)
        timestep_constrained_viterbi_path = self.timestep_constrained_crf.viterbi_tags(self.logits, mask)

        # Separate the tags and scores.
        viterbi_tags = [x for x, y in viterbi_path]
        viterbi_scores = [y for x, y in viterbi_path]
        timestep_constrained_viterbi_tags = [x for x, y in timestep_constrained_viterbi_path]
        timestep_constrained_viterbi_scores = [y for x, y in timestep_constrained_viterbi_path]

        # Check that the viterbi tags are what I think they should be.
        assert viterbi_tags == [
                [2, 4, 3],
                [4, 2]
        ]
        assert timestep_constrained_viterbi_tags == [
                [2, 4, 3],
                [4, 2]
        ]

        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []
        best_scores = []

        for logit, mas in zip(self.logits, mask):
            sequence_length = torch.sum(mas.detach())
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        assert viterbi_tags == most_likely_tags
        assert viterbi_scores == best_scores
        assert timestep_constrained_viterbi_tags == most_likely_tags
        assert timestep_constrained_viterbi_scores == best_scores

    def test_constrained_viterbi_tags(self):
        constraints = {(0, 0), (0, 1),
                       (1, 1), (1, 2),
                       (2, 2), (2, 3),
                       (3, 3), (3, 4),
                       (4, 4), (4, 0)}

        # Add the transitions to the end tag
        # and from the start tag.
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))

        crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        crf.transitions = torch.nn.Parameter(self.transitions)
        crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        timestep_constrained_crf = TimestepConstrainedConditionalRandomField(num_tags=5, constraints=constraints)
        timestep_constrained_crf.transitions = torch.nn.Parameter(self.transitions)
        timestep_constrained_crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        timestep_constrained_crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        mask = torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ])

        viterbi_path = crf.viterbi_tags(self.logits, mask)
        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = [x for x, y in viterbi_path]
        # Now the tags should respect the constraints
        assert viterbi_tags == [
                [2, 3, 3],
                [2, 3]
        ]

        timestep_constrained_viterbi_path = timestep_constrained_crf.viterbi_tags(self.logits, mask)
        # Get just the tags from each tuple of (tags, score).
        timestep_constrained_viterbi_tags = [x for x, y in timestep_constrained_viterbi_path]
        # Now the tags should respect the constraints
        assert timestep_constrained_viterbi_tags == viterbi_tags

        timestep_constrained_crf_no_constraints = TimestepConstrainedConditionalRandomField(num_tags=5)
        timestep_constrained_crf_no_constraints.transitions = torch.nn.Parameter(self.transitions)
        timestep_constrained_crf_no_constraints.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        timestep_constrained_crf_no_constraints.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        constraint_mask = crf._constraint_mask.data
        expanded_constraint_mask = constraint_mask.unsqueeze(0).expand(2, -1, -1)
        # Pass in the constraints at viterbi_tags
        timestep_constrained_viterbi_path = timestep_constrained_crf_no_constraints.viterbi_tags(
            self.logits, mask, expanded_constraint_mask)
        # Get just the tags from each tuple of (tags, score).
        timestep_constrained_viterbi_tags = [x for x, y in timestep_constrained_viterbi_path]
        # Now the tags should respect the constraints
        assert timestep_constrained_viterbi_tags == viterbi_tags
