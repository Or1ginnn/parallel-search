import unittest

import torch

from verl.trainer.ppo.core_algos import (
    kl_penalty,
    pad_trajectory_tensors_for_concat,
    select_informative_group_indices,
)


class LowVarianceKLPenaltyTest(unittest.TestCase):

    def test_matches_formula_in_normal_range(self):
        logprob = torch.tensor([-3.0, -2.0, -1.0], requires_grad=True)
        ref_logprob = torch.tensor([-2.5, -2.0, -1.5])

        actual = kl_penalty(logprob, ref_logprob, "low_var_kl")
        delta = ref_logprob - logprob
        expected = torch.exp(delta) - delta - 1.0

        torch.testing.assert_close(actual, expected)

    def test_extreme_logprob_gap_has_finite_backward(self):
        logprob = torch.tensor([-1000.0, 0.0, -100.0, 0.0], requires_grad=True)
        ref_logprob = torch.tensor([0.0, -1000.0, 0.0, -100.0])

        penalty = kl_penalty(logprob, ref_logprob, "low_var_kl")
        penalty.sum().backward()

        self.assertTrue(torch.isfinite(penalty).all())
        self.assertTrue(torch.isfinite(logprob.grad).all())
        torch.testing.assert_close(penalty, torch.full_like(penalty, 10.0))

    def test_bfloat16_inputs_are_computed_in_float32(self):
        logprob = torch.tensor([-1000.0, 0.0], dtype=torch.bfloat16, requires_grad=True)
        ref_logprob = torch.tensor([0.0, -1000.0], dtype=torch.bfloat16)

        penalty = kl_penalty(logprob, ref_logprob, "low_var_kl")
        penalty.sum().backward()

        self.assertEqual(penalty.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logprob.grad).all())


class DynamicGroupSamplingTest(unittest.TestCase):

    def test_filters_zero_variance_groups_and_preserves_complete_groups(self):
        indices, group_ids, group_stds = select_informative_group_indices(
            index=["q1", "q1", "q2", "q2", "q3", "q3"],
            scores=[0.0, 1.0, 0.0, 0.0, 0.2, 0.4],
        )

        self.assertEqual(group_ids, ["q1", "q3"])
        self.assertEqual(indices.tolist(), [0, 1, 4, 5])
        self.assertGreater(group_stds["q1"], 0.0)
        self.assertEqual(group_stds["q2"], 0.0)

    def test_caps_selection_by_whole_group(self):
        indices, group_ids, _ = select_informative_group_indices(
            index=["q1", "q1", "q2", "q2"],
            scores=[0.0, 1.0, 0.0, 1.0],
            max_groups=1,
        )

        self.assertEqual(group_ids, ["q1"])
        self.assertEqual(indices.tolist(), [0, 1])

    def test_pads_variable_rollout_lengths_before_concat(self):
        short_batch = {
            "prompts": torch.tensor([[0, 11, 12]]),
            "responses": torch.tensor([[21, 22]]),
            "input_ids": torch.tensor([[0, 11, 12, 21, 22]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3]]),
            "info_mask": torch.tensor([[0, 1, 1, 1, 1]]),
            "token_level_scores": torch.tensor([[0.0, 0.25]]),
        }
        long_batch = {
            "prompts": torch.tensor([[11, 12, 13]]),
            "responses": torch.tensor([[21, 22, 23]]),
            "input_ids": torch.tensor([[11, 12, 13, 21, 22, 23]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "position_ids": torch.arange(6).unsqueeze(0),
            "info_mask": torch.ones(1, 6, dtype=torch.long),
            "token_level_scores": torch.tensor([[0.0, 0.0, 1.25]]),
        }

        pad_trajectory_tensors_for_concat([short_batch, long_batch], pad_token_id=0)

        self.assertEqual(short_batch["responses"].tolist(), [[21, 22, 0]])
        self.assertEqual(short_batch["input_ids"].tolist(), [[0, 11, 12, 21, 22, 0]])
        self.assertEqual(short_batch["attention_mask"].tolist(), [[0, 1, 1, 1, 1, 0]])
        self.assertEqual(short_batch["position_ids"].tolist(), [[0, 0, 1, 2, 3, 0]])
        self.assertEqual(short_batch["token_level_scores"].tolist(), [[0.0, 0.25, 0.0]])
        for key in short_batch:
            torch.cat([short_batch[key], long_batch[key]], dim=0)


if __name__ == "__main__":
    unittest.main()
