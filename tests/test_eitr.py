import unittest
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "ppo" / "eitr.py"
SPEC = importlib.util.spec_from_file_location("eitr_module", MODULE_PATH)
EITR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EITR)

build_sibling_probe_tensors = EITR.build_sibling_probe_tensors
build_online_probe_tensors = EITR.build_online_probe_tensors
induced_js_from_cached_effects = EITR.induced_js_from_cached_effects
update_dual_beta = EITR.update_dual_beta
validate_eitr_config = EITR.validate_eitr_config
validate_sibling_group_layout = EITR.validate_sibling_group_layout


class EITRMathTest(unittest.TestCase):
    def test_induced_js_is_zero_at_old_policy_and_has_finite_gradient(self):
        old = torch.zeros(1, 4)
        current = old.clone().requires_grad_(True)
        docs = torch.eye(4).unsqueeze(0)
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = induced_js_from_cached_effects(
            current_seq_logp=current,
            old_seq_logp=old,
            doc_probs=docs,
            probe_mask=mask,
        )
        self.assertAlmostEqual(result["js"].item(), 0.0, places=7)
        result["js"].sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())

        shifted = torch.tensor([[2.0, -2.0, -2.0, -2.0]], requires_grad=True)
        shifted_result = induced_js_from_cached_effects(
            current_seq_logp=shifted,
            old_seq_logp=old,
            doc_probs=docs,
            probe_mask=mask,
        )
        self.assertGreater(shifted_result["js"].item(), 0.0)
        shifted_result["js"].sum().backward()
        self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_dual_update_respects_target_and_bounds(self):
        self.assertAlmostEqual(update_dual_beta(0.1, 0.03, 0.01, 0.5, 10.0), 0.11)
        self.assertEqual(update_dual_beta(0.0, 0.0, 1.0, 1.0, 10.0), 0.0)
        self.assertEqual(update_dual_beta(9.9, 1.0, 0.0, 1.0, 10.0), 10.0)


class EITRProbeBatchTest(unittest.TestCase):
    @staticmethod
    def _record(action_ids, doc_prefix):
        return {
            "queries": [f"query {doc_prefix}"],
            "prefix_text": "",
            "action_token_ids": action_ids,
            "retrieval_effect": [
                {"doc_id": f"{doc_prefix}-a", "score": 1.0},
                {"doc_id": f"{doc_prefix}-b", "score": 0.5},
            ],
        }

    def test_builds_representative_probe_groups_and_dummy_slots(self):
        batch_size = 10
        prompt_width = 4
        response_width = 6
        prompts = torch.tensor([[0, 11, 12, 13]] * batch_size)
        attention_mask = torch.ones(batch_size, prompt_width + response_width, dtype=torch.long)
        responses = torch.zeros(batch_size, response_width, dtype=torch.long)
        old_log_probs = torch.full((batch_size, response_width), -0.25)
        records = []
        for index in range(batch_size):
            action_ids = [21 + index, 31 + index]
            responses[index, :2] = torch.tensor(action_ids)
            records.append(self._record(action_ids, f"d{index}"))

        # The second uid group has only three eligible search actions.
        records[8] = None
        records[9] = None
        uids = ["q0"] * 5 + ["q1"] * 5
        tensors, metrics = build_sibling_probe_tensors(
            prompts=prompts,
            attention_mask=attention_mask,
            responses=responses,
            old_log_probs=old_log_probs,
            uids=uids,
            records=records,
            pad_token_id=0,
            config={
                "probe_count": 4,
                "max_action_tokens": 8,
                "max_doc_support": 16,
                "retrieval_score_temperature": 0.1,
            },
        )

        self.assertEqual(tensors["eitr_state_slot"].nonzero().flatten().tolist(), [0, 5])
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0])
        self.assertEqual(int(tensors["eitr_probe_valid"][0].sum()), 4)
        self.assertEqual(int(tensors["eitr_probe_valid"][5].sum()), 0)
        self.assertAlmostEqual(metrics["eitr/probe_state_coverage"], 0.5)
        self.assertTrue(torch.allclose(tensors["eitr_probe_doc_probs"][0].sum(dim=-1), torch.ones(4)))

    def test_layout_and_config_guards(self):
        validate_sibling_group_layout(["a"] * 5 + ["b"] * 5, n_agent=5, world_size=2)
        with self.assertRaises(ValueError):
            validate_sibling_group_layout(["a", "b"] * 5, n_agent=5, world_size=2)
        validate_eitr_config({}, n_agent=5, max_queries_per_turn=1, rollout_n=1)
        with self.assertRaises(ValueError):
            validate_eitr_config({}, n_agent=3, max_queries_per_turn=1, rollout_n=1)

    def test_online_probes_share_one_fixed_state(self):
        prompts = torch.tensor([[0, 11, 12, 13]] * 5)
        attention_mask = torch.ones(5, 10, dtype=torch.long)
        responses = torch.tensor([[51, 52, 0, 0, 0, 0]] * 5)
        probes = []
        for index in range(4):
            probes.append({
                "query": f"q{index}",
                "action_token_ids": [60 + index, 70],
                "retrieval_effect": [{"doc_id": f"doc-{index}", "score": 1.0}],
            })
        groups = [{
            "state_prompt_token_ids": [11, 12, 13, 50],
            "extra_retrieval_calls": 3,
            "probes": probes,
        }] + [None] * 4
        tensors, metrics = build_online_probe_tensors(
            prompts=prompts,
            attention_mask=attention_mask,
            responses=responses,
            uids=["q0"] * 5,
            probe_groups=groups,
            pad_token_id=0,
            config={"probe_count": 4, "max_action_tokens": 8, "max_doc_support": 8},
        )
        self.assertEqual(metrics["eitr/probe_state_coverage"], 1.0)
        self.assertEqual(metrics["eitr/probe_retrieval_call_count"], 3.0)
        self.assertEqual(tensors["eitr_state_slot"].nonzero().flatten().tolist(), [0])
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0])
        state_prefixes = tensors["eitr_probe_input_ids"][0, :, :4]
        self.assertTrue(torch.equal(state_prefixes, state_prefixes[0].expand_as(state_prefixes)))


if __name__ == "__main__":
    unittest.main()
