import unittest

import torch

from pa_moelog.models import PAMoELog, ParameterEncoder


def _parameters(ip: str):
    return [{"IP": [ip], "PATH": [], "URL": [], "HEX": [], "NUM": [], "PORT": [],
             "USER": [], "PID": [], "FILE": [], "TIME": []}]


class ParameterAttentionTest(unittest.TestCase):
    def test_parameter_encoder_preserves_individual_tokens_and_mask(self):
        encoder = ParameterEncoder(hidden_dim=8)
        parameters = [_parameters("10.0.0.1")[0], {**_parameters("10.0.0.2")[0], "PORT": ["22"]}]
        tokens, mask = encoder.encode_tokens(parameters)

        self.assertEqual(tokens.shape, (2, 2, 8))
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        self.assertFalse(torch.allclose(tokens[1, 0], tokens[1, 1]))

    def test_parameter_change_affects_shared_representation(self):
        torch.manual_seed(7)
        model = PAMoELog(hidden_dim=16, num_experts=2, backbone_name="simple-hash-encoder")
        model.eval()
        text = ["connection failed from <IP>"]

        with torch.no_grad():
            first = model(text, _parameters("10.0.0.1"))["shared_hidden"]
            second = model(text, _parameters("10.0.0.2"))["shared_hidden"]

        self.assertFalse(torch.allclose(first, second))

    def test_empty_parameters_produce_finite_output(self):
        model = PAMoELog(hidden_dim=16, num_experts=2, backbone_name="simple-hash-encoder")
        empty = [{name: [] for name in ("IP", "PATH", "URL", "HEX", "NUM", "PORT", "USER", "PID", "FILE", "TIME")}]
        with torch.no_grad():
            output = model(["service started"], empty)
        self.assertTrue(torch.isfinite(output["final_score"]).all())


if __name__ == "__main__":
    unittest.main()
