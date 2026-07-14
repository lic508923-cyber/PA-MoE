import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from pa_moelog.data import LogSequenceDataset
from pa_moelog.models import LightweightExpertFusion,PAMoELog,ParameterEncoder
from pa_moelog.utils import load_checkpoint,save_checkpoint
from scripts.evaluate import validate_checkpoint_mode
from scripts.train_multisource import SystemBalancedBatchSampler

EMPTY={name:[] for name in ("IP","PATH","URL","HEX","NUM","PORT","USER","PID","FILE","TIME")}

class SequenceAndTrainingSafetyTest(unittest.TestCase):
    def test_event_order_changes_sequence_representation(self):
        torch.manual_seed(4); model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); model.eval()
        with torch.no_grad():
            forward=model.encode_sequences([["start job","fail job"]],[[EMPTY,EMPTY]])
            reverse=model.encode_sequences([["fail job","start job"]],[[EMPTY,EMPTY]])
        self.assertFalse(torch.allclose(forward,reverse))

    def test_text_padding_is_masked_from_pooling(self):
        torch.manual_seed(5); model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); model.eval()
        parameter,parameter_mask=model.parameter_encoder.encode_tokens([EMPTY])
        real=torch.randn(1,2,16); padded=torch.cat([real,torch.randn(1,3,16)],dim=1)
        with torch.no_grad():
            short=model.fusion_encoder(real,parameter,parameter_mask,torch.tensor([[True,True]]))
            long=model.fusion_encoder(padded,parameter,parameter_mask,torch.tensor([[True,True,False,False,False]]))
        self.assertTrue(torch.allclose(short,long,atol=1e-6))

    def test_actual_encoder_representation_ignores_extra_padding(self):
        torch.manual_seed(9); model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); model.eval()
        model.text_encoder.max_length=16
        with torch.no_grad(): short=model.encode_events(["same short log"],[EMPTY])
        model.text_encoder.max_length=32
        with torch.no_grad(): long=model.encode_events(["same short log"],[EMPTY])
        self.assertTrue(torch.allclose(short,long,atol=1e-6))

    def test_only_one_parameter_encoder_is_registered(self):
        model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder")
        self.assertEqual(sum(isinstance(module,ParameterEncoder) for module in model.modules()),1)

    def test_timestamp_sorting_and_mixed_sessions_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"events.csv"
            path.write_text("log,label,system,timestamp\nsecond,0,A,2\nfirst,0,A,1\n",encoding="utf-8")
            dataset=LogSequenceDataset(path,window_size=2)
            self.assertEqual(dataset[0]["raw_logs"],["first","second"])
            path.write_text("log,label,system,timestamp,session_id\na,0,A,1,s1\nb,0,A,2,\n",encoding="utf-8")
            with self.assertRaises(ValueError): LogSequenceDataset(path,window_size=2)

    def test_strict_split_keeps_sessions_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source.csv"; output=root/"splits"
            with source.open("w",encoding="utf-8",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["log","label","system","timestamp","session_id"]); writer.writeheader()
                for index in range(10): writer.writerow({"log":f"e{index}","label":index%2,"system":"A","timestamp":index,"session_id":f"s{index}"})
            script=Path(__file__).resolve().parents[1]/"scripts"/"prepare_splits.py"
            subprocess.run([sys.executable,str(script),"--input-csv",str(source),"--output-dir",str(output),"--window-size","2"],check=True,capture_output=True,text=True)
            seen=set()
            for name in ("train","support","validation","test"):
                with (output/f"{name}.csv").open(encoding="utf-8") as handle:
                    current={row["session_id"] for row in csv.DictReader(handle)}
                self.assertFalse(seen & current); seen |= current
            self.assertEqual(len(seen),10)
            self.assertTrue((output/"train_sequences.csv").exists())

    def test_prepare_splits_builds_windows_only_inside_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source.csv"; output=root/"splits"
            with source.open("w",encoding="utf-8",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["log","label","system","timestamp"]); writer.writeheader()
                for index in range(20): writer.writerow({"log":f"unique-{index}","label":index%2,"system":"A","timestamp":index})
            script=Path(__file__).resolve().parents[1]/"scripts"/"prepare_splits.py"
            subprocess.run([sys.executable,str(script),"--input-csv",str(source),"--output-dir",str(output),
                            "--window-size","2","--stride","2"],check=True,capture_output=True,text=True)
            raw_seen=set()
            for name in ("train","support","validation","test"):
                with (output/f"{name}.csv").open(encoding="utf-8") as handle:
                    raw={row["log"] for row in csv.DictReader(handle)}
                self.assertFalse(raw_seen & raw); raw_seen |= raw
                with (output/f"{name}_sequences.csv").open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle): self.assertIn(row["log"],raw)
            self.assertEqual(len(raw_seen),20)

    def test_fusion_shrinks_to_uniform_when_label_budget_is_small(self):
        fusion=LightweightExpertFusion(2,shrinkage_strength=10)
        fusion.calibrate_from_distances(torch.tensor([0.0,10.0]),label_budget=0)
        self.assertTrue(torch.allclose(fusion.weights,torch.tensor([.5,.5])))
        fusion.calibrate_from_distances(torch.tensor([0.0,10.0]),label_budget=1000)
        self.assertGreater(float(fusion.weights[0]),.98)

    def test_checkpoint_loading_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.pt"; model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder")
            save_checkpoint(path,model,{})
            checkpoint=torch.load(path); checkpoint["model_state_dict"].pop("energy_scale"); torch.save(checkpoint,path)
            with self.assertRaises(RuntimeError): load_checkpoint(path,model=model)

    def test_checkpoint_roundtrip_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.pt"; torch.manual_seed(12)
            model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); model.eval()
            with torch.no_grad(): expected=model(["service start"],[EMPTY])["logit"]
            save_checkpoint(path,model,{"sequence":False})
            restored=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); restored.eval()
            load_checkpoint(path,model=restored)
            with torch.no_grad(): actual=restored(["service start"],[EMPTY])["logit"]
            self.assertTrue(torch.equal(expected,actual))

    def test_whole_model_is_batch_invariant(self):
        torch.manual_seed(13); model=PAMoELog(hidden_dim=16,num_experts=1,backbone_name="simple-hash-encoder"); model.eval()
        with torch.no_grad():
            single=model(["same event"],[EMPTY])["logit"]
            batch=model(["same event","unrelated event"],[EMPTY,EMPTY])["logit"][:1]
        self.assertTrue(torch.allclose(single,batch,atol=1e-6))

    def test_sequence_checkpoint_cannot_use_event_evaluation(self):
        with self.assertRaises(ValueError): validate_checkpoint_mode({"sequence":True},False)
        validate_checkpoint_mode({"sequence":True},True)

    def test_balanced_batch_sampler_includes_all_systems_when_feasible(self):
        rows=[{"system":"A"} for _ in range(20)]+[{"system":"B"}]
        sampler=SystemBalancedBatchSampler(rows,batch_size=4,seed=1)
        first=next(iter(sampler)); systems={rows[index]["system"] for index in first}
        self.assertEqual(systems,{"A","B"})

    def test_multisource_toy_batch_can_be_overfit(self):
        torch.manual_seed(21); model=PAMoELog(hidden_dim=8,num_experts=2,backbone_name="simple-hash-encoder")
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout): module.p=0.0
        texts=[["alpha normal"],["alpha fatal"],["beta normal"],["beta panic"]]
        parameters=[[EMPTY] for _ in texts]; labels=torch.tensor([0.,1.,0.,1.]); expert_ids=torch.tensor([0,0,1,1])
        optimizer=torch.optim.Adam(model.parameters(),lr=.005)
        for _ in range(200):
            hidden=model.encode_sequences(texts,parameters)
            logits=torch.stack([expert(hidden)["logit"] for expert in model.expert_pool.experts],1)
            selected=logits.gather(1,expert_ids[:,None]).squeeze(1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(selected,labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()),.05)

if __name__=="__main__": unittest.main()
