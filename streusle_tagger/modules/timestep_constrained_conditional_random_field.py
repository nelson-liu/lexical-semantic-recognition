from typing import List, Tuple

from allennlp.modules import ConditionalRandomField
import allennlp.nn.util as util
import torch


class TimestepConstrainedConditionalRandomField(ConditionalRandomField):
    """
    This module modifies the `allennlp.modules.ConditionalRandomField` to
    take constraints for CRF decoding for individual instances, instead of
    using the same set of constraints for all instances.
    """
    def viterbi_tags(self,
                     logits: torch.Tensor,
                     mask: torch.Tensor,
                     constraint_mask: torch.Tensor = None) -> List[Tuple[List[int], float]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Parameters
        ----------
        logits: torch.Tensor
            Shape: (batch_size, max_seq_length, num_tags) Tensor of logits.
        mask: torch.Tensor
            Shape: (batch_size, max_seq_length, num_tags) Tensor of logits.
        constraint_mask: torch.Tensor, optional (default=None)
            Shape: (batch_size, num_tags+2, num_tags+2) Tensor of the allowed
            transitions for each example in the batch.
        """
        # pylint: disable=arguments-differ
        if constraint_mask is None:
            # Defer to superclass function if there is no custom constraint mask.
            return super().viterbi_tags(logits=logits,
                                        mask=mask)
        # We have a custom constraint mask for each example, so we need to re-mask
        # when we make each prediction.
        batch_size, max_seq_length, num_tags = logits.size()

        assert list(constraint_mask.size()) == [batch_size, num_tags + 2, num_tags + 2]

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        start_tag = num_tags
        end_tag = num_tags + 1
        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask, prediction_constraint_mask in zip(logits, mask, constraint_mask):
            prediction_constraint_mask = torch.nn.Parameter(prediction_constraint_mask, requires_grad=False)
            # Augment transitions matrix with start and end transitions
            transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)
            # Apply transition constraints
            constrained_transitions = (
                    self.transitions * prediction_constraint_mask[:num_tags, :num_tags] +
                    -10000.0 * (1 - prediction_constraint_mask[:num_tags, :num_tags])
            )
            transitions[:num_tags, :num_tags] = constrained_transitions.data

            if self.include_start_end_transitions:
                transitions[start_tag, :num_tags] = (
                        self.start_transitions.detach() * prediction_constraint_mask[start_tag, :num_tags].data +
                        -10000.0 * (1 - prediction_constraint_mask[start_tag, :num_tags].detach())
                )
                transitions[:num_tags, end_tag] = (
                        self.end_transitions.detach() * prediction_constraint_mask[:num_tags, end_tag].data +
                        -10000.0 * (1 - prediction_constraint_mask[:num_tags, end_tag].detach())
                )
            else:
                transitions[start_tag, :num_tags] = (
                        -10000.0 * (1 - prediction_constraint_mask[start_tag, :num_tags].detach()))
                transitions[:num_tags, end_tag] = (
                        -10000.0 * (1 - prediction_constraint_mask[:num_tags, end_tag].detach()))

            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))
        return best_paths
