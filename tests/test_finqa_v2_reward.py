import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid importing verl.__init__ in lightweight local test environments where
# the full training-only dependency stack is intentionally unavailable.
for package_name in ("verl", "verl.utils", "verl.utils.reward_score"):
    sys.modules.setdefault(package_name, types.ModuleType(package_name))

_load_module("verl.utils.reward_score.qa_em", "verl/utils/reward_score/qa_em.py")
finqa_metrics = _load_module(
    "verl.utils.reward_score.finqa_metrics",
    "verl/utils/reward_score/finqa_metrics.py",
)
litecoa_qa = _load_module(
    "verl.utils.reward_score.litecoa_qa",
    "verl/utils/reward_score/litecoa_qa.py",
)


class FinQAV2RewardTest(unittest.TestCase):

    def setUp(self):
        self.ground_truth = {
            "target": ["11%"],
            "answer_type": "finqa",
            "program": "subtract(700, 600), divide(#0, 600), multiply(#1, const_100)",
            "executable_answer": "0.10559",
            "gold_evidence": [],
        }

    def test_percent_and_ratio_are_equivalent_for_error(self):
        self.assertEqual(finqa_metrics.numeric_relative_error("0.935", "93.5%"), 0.0)

    def test_original_reward_api_is_unchanged_when_v2_is_disabled(self):
        score = litecoa_qa.compute_score_em_litecoa(
            "<search>capital || france</search><answer>Paris</answer>",
            "<information>[Query] capital\nParis is the capital of France.</information>",
            {"target": ["Paris"]},
            answer_present_bonus=0.05,
            no_generated_information_bonus=0.05,
            evidence_hit_bonus=0.05,
            valid_search_bonus=0.03,
            parallel_evidence_bonus=0.03,
        )
        self.assertAlmostEqual(score, 1.21)

    def test_program_operands_ignore_constants_and_step_references(self):
        self.assertEqual(litecoa_qa._program_operands(self.ground_truth["program"]), [700.0, 600.0])

    def test_parallel_gain_requires_complementary_evidence(self):
        response = "<think>x</think><search>q700 || q600</search><think>x</think><answer>16.67%</answer>"
        information = """<information>
[Query] q700
Doc 1 revenue in 2008 was 700.
[Query] q600
Doc 1 revenue in 2007 was 600.
</information>"""

        components = litecoa_qa.compute_score_em_litecoa_components(
            response,
            information,
            self.ground_truth,
            answer_present_bonus=0.05,
            no_generated_information_bonus=0.05,
            valid_search_bonus=0.05,
            finqa_v2_reward=True,
        )

        self.assertEqual(components["answer_em"], 0.0)
        self.assertEqual(components["retrieval_coverage"], 1.0)
        self.assertEqual(components["parallel_retrieval_gain"], 0.5)
        self.assertAlmostEqual(components["score"], 0.225)

    def test_duplicate_parallel_queries_have_no_marginal_gain(self):
        response = "<search>q1 || q2</search><answer>16.67%</answer>"
        information = """<information>
[Query] q1
Doc 1 values were 700 and 600.
[Query] q2
Doc 1 values were 700 and 600.
</information>"""
        gain = litecoa_qa._parallel_retrieval_gain(response, information, self.ground_truth)
        self.assertEqual(gain, 0.0)

    def test_query_text_cannot_fake_retrieval_coverage(self):
        information = """<information>
[Query] find 700 and 600
Doc 1 contains no relevant financial values.
</information>"""
        self.assertEqual(litecoa_qa._retrieval_coverage(information, self.ground_truth), 0.0)

    def test_near_miss_reward_is_small_and_continuous(self):
        ground_truth = {
            "target": ["100"],
            "answer_type": "finqa",
            "program": "subtract(120, 20)",
            "executable_answer": "100.0",
            "gold_evidence": [],
        }
        response = "<search>values</search><answer>103</answer>"
        components = litecoa_qa.compute_score_em_litecoa_components(
            response,
            "<information>[Query] values\nDoc 1 values were 120 and 20.</information>",
            ground_truth,
            answer_present_bonus=0.05,
            no_generated_information_bonus=0.05,
            valid_search_bonus=0.05,
            finqa_v2_reward=True,
        )

        self.assertAlmostEqual(components["numeric_near_miss_quality"], 0.5)
        self.assertAlmostEqual(components["score"], 0.225)

    def test_wrong_scale_does_not_receive_near_miss_reward(self):
        quality = litecoa_qa._numeric_near_miss_quality(
            "<answer>9350%</answer>",
            {
                "target": ["93.5%"],
                "answer_type": "finqa",
                "executable_answer": "0.935",
            },
            max_relative_error=0.05,
        )
        self.assertEqual(quality, 0.0)


if __name__ == "__main__":
    unittest.main()
