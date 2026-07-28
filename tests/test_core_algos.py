import unittest

import torch

from verl.trainer.ppo.core_algos import kl_penalty


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


if __name__ == "__main__":
    unittest.main()
