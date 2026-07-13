import tempfile
import unittest
from pathlib import Path
import torch
from pa_moelog.data import LogSequenceDataset, collate_fn
from pa_moelog.models import DoRALinear, GMMEnergy, LightweightExpertFusion, PAMoELog

class ArchitectureClosureTest(unittest.TestCase):
    def test_dora_is_identity_at_initialization(self):
        layer=DoRALinear(8,8); x=torch.randn(3,8)
        self.assertTrue(torch.allclose(layer(x),x,atol=1e-6))

    def test_gmm_components_are_fitted_and_distinct(self):
        torch.manual_seed(1); hidden=torch.cat([torch.randn(20,4)-3,torch.randn(20,4)+3])
        gmm=GMMEnergy(4,2); gmm.fit_normal(hidden)
        self.assertTrue(bool(gmm.is_fitted)); self.assertFalse(torch.allclose(gmm.means[0],gmm.means[1]))

    def test_untrained_expert_is_excluded(self):
        fusion=LightweightExpertFusion(3); fusion.set_trained_mask(torch.tensor([True,False,True]))
        fusion.calibrate_from_distances(torch.tensor([1.0,0.0,2.0]))
        self.assertEqual(float(fusion.weights[1]),0.0)

    def test_sequence_dataset_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"logs.csv"
            path.write_text("log,label,system,timestamp\nstart,0,A,1\nfailed port=22,1,A,2\n",encoding="utf-8")
            dataset=LogSequenceDataset(path,window_size=2,stride=2); batch=collate_fn([dataset[0]])
            model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder")
            out=model(batch["semantic_texts"],batch["parameters"],batch["event_mask"])
            self.assertEqual(out["shared_hidden"].shape,(1,16))

    def test_energy_score_is_batch_invariant_after_fit(self):
        torch.manual_seed(3); model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder")
        normal=torch.randn(8,16); model.gmm_energy.fit_normal(normal); model.fit_energy_statistics(normal)
        one=model._normalize_energy(model.gmm_energy(normal[:1])["energy"])
        many=model._normalize_energy(model.gmm_energy(normal[:4])["energy"])
        self.assertTrue(torch.allclose(one,many[:1]))

if __name__ == "__main__": unittest.main()
