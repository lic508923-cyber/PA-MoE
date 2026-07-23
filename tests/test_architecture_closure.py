import tempfile
import unittest
from pathlib import Path
import torch
from pa_moelog.data import LogSequenceDataset, collate_fn
from pa_moelog.models import (DoRALinear, DoRAWeightParametrization, GMMEnergy,
                              LightweightExpertFusion, PAMoELog,
                              TargetConditionedExpertGate)
from torch.nn.utils import parametrize

class ArchitectureClosureTest(unittest.TestCase):
    def test_dora_is_identity_at_initialization(self):
        layer=DoRALinear(8,8); x=torch.randn(3,8)
        self.assertTrue(torch.allclose(layer(x),x,atol=1e-6))

    def test_dora_copies_a_trained_rectangular_linear_exactly(self):
        torch.manual_seed(4); source=torch.nn.Linear(5,8); layer=DoRALinear(5,8,rank=2,alpha=2)
        layer.load_base_from_linear(source); x=torch.randn(3,5)
        torch.testing.assert_close(layer(x),source(x),rtol=1e-6,atol=1e-6)

    def test_weight_parametrization_preserves_existing_linear(self):
        torch.manual_seed(41); layer=torch.nn.Linear(7,5); x=torch.randn(3,7)
        expected=layer(x)
        parametrize.register_parametrization(
            layer,"weight",DoRAWeightParametrization(layer.weight.detach(),rank=3,alpha=3))
        torch.testing.assert_close(layer(x),expected,rtol=1e-6,atol=1e-6)

    def test_target_gate_starts_from_static_prior_and_is_sample_conditioned(self):
        gate=TargetConditionedExpertGate(4,3); hidden=torch.tensor([[1.,0,0,0],[-1.,0,0,0]])
        prior=torch.tensor([.2,.3,.5]); mask=torch.tensor([True,True,True])
        torch.testing.assert_close(gate(hidden,prior,mask),prior.expand(2,-1))
        with torch.no_grad(): gate.projection.weight[0,0]=2
        weights=gate(hidden,prior,mask)
        self.assertFalse(torch.allclose(weights[0],weights[1]))
        torch.testing.assert_close(weights.sum(1),torch.ones(2))

    def test_expert_dora_preserves_source_experts_when_enabled(self):
        torch.manual_seed(5); model=PAMoELog(hidden_dim=16,num_experts=3,
            backbone_name="simple-hash-encoder"); model.eval(); shared=torch.randn(4,16)
        static=model.fusion(4,device=shared.device,dtype=shared.dtype)
        with torch.no_grad(): source=model.expert_pool(shared,static)
        model.enable_expert_dora(rank=4,alpha=4); model.eval()
        with torch.no_grad(): adapted=model.expert_pool(shared,static,target_adapted=True)
        torch.testing.assert_close(adapted["expert_hiddens"],source["expert_hiddens"])
        torch.testing.assert_close(adapted["expert_logits"],source["expert_logits"])

    def test_deep_dora_preserves_complete_source_output_when_enabled(self):
        torch.manual_seed(42); model=PAMoELog(hidden_dim=16,num_experts=3,
            backbone_name="simple-hash-encoder"); model.eval()
        empty={key:[] for key in ("IP","PATH","URL","HEX","NUM","PORT","USER","PID","FILE","TIME")}
        with torch.no_grad(): source=model(["service event"],[empty])
        model.enable_deep_dora(rank=4,alpha=4); model.eval()
        with torch.no_grad(): adapted=model(["service event"],[empty])
        torch.testing.assert_close(adapted["logit"],source["logit"])
        torch.testing.assert_close(adapted["expert_logits"],source["expert_logits"])

    def test_gmm_components_are_fitted_and_distinct(self):
        torch.manual_seed(1); hidden=torch.cat([torch.randn(20,4)-3,torch.randn(20,4)+3])
        gmm=GMMEnergy(4,2); gmm.fit_normal(hidden)
        self.assertTrue(bool(gmm.is_fitted)); self.assertEqual(int(gmm.active_components),2)
        self.assertFalse(torch.allclose(gmm.means[0],gmm.means[1]))

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
        self.assertEqual(int(model.gmm_energy.active_components),1)
        self.assertLessEqual(int(model.gmm_energy.active_projection_dim),7)
        self.assertGreaterEqual(float(model.energy_scale),0.1)
        one=model._normalize_energy(model.gmm_energy(normal[:1])["energy"])
        many=model._normalize_energy(model.gmm_energy(normal[:4])["energy"])
        torch.testing.assert_close(one,many[:1],rtol=1e-5,atol=1e-5)

if __name__ == "__main__": unittest.main()
