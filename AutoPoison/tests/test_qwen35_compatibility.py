import unittest
from types import SimpleNamespace
from AutoPoison.model_adapters.qwen35 import QWEN35_MODEL_ID, classify_parameter, is_qwen35_config
from AutoPoison.probes.qwen35_memory_probe import parser, STAGES
class P:
 def __init__(self,shape=(2,2)): self.ndim=len(shape); self.shape=shape
class TestQwen35Contracts(unittest.TestCase):
 def test_registration(self): self.assertEqual(QWEN35_MODEL_ID,"Qwen/Qwen3.5-4B-Base")
 def test_config_recognition(self): self.assertTrue(is_qwen35_config(SimpleNamespace(model_type="qwen3_5",architectures=[])))
 def test_vision_and_delta_classification(self): self.assertEqual(classify_parameter("model.visual.blocks.0.weight",P())[1],"VISION_MODULE"); self.assertEqual(classify_parameter("model.layers.0.deltanet.in_proj.weight",P())[0],"candidate")
 def test_matrix_and_unknown(self): self.assertEqual(classify_parameter("model.layers.0.foo.weight",P())[0],"unknown"); self.assertEqual(classify_parameter("model.layers.0.mlp.down_proj.weight",P((4,)))[1],"NON_MATRIX_PARAMETER")
 def test_one_stage_validation(self): self.assertEqual(set(STAGES),{"load","forward","backward","optimizer-init","optimizer-step","constraint-build","projection"}); self.assertEqual(parser().parse_args(["--model-id","x","--output","o","--stage","load"]).stage,"load")
if __name__ == "__main__": unittest.main()
