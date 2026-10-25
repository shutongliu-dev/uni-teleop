import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_sonic_action import REFERENCE_ACTION
from test_sonic_observation import robot_state

from sonic_mujoco.controllers.sonic import OnnxPolicy, SonicObservationBuilder

REFERENCE_ROOT = Path(os.environ.get("SONIC_REFERENCE_ROOT", "/nonexistent"))
DEPLOY = REFERENCE_ROOT / "gear_sonic_deploy"
MODEL = DEPLOY / "policy/release/model_decoder.onnx"
INCLUDE = DEPLOY / "src/g1/g1_deploy_onnx_ref/include"
CUDA = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))
TENSORRT = Path(os.environ.get("TENSORRT_ROOT", "/nonexistent"))
TRT_LIBRARY = DEPLOY / "build/src/TRTInference/libTRTInference.a"


class SonicEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (MODEL, CUDA, TENSORRT, TRT_LIBRARY)
        if not all(path.exists() for path in required) or shutil.which("g++") is None:
            raise unittest.SkipTest("SONIC C++ reference is not available")
        source = Path(__file__).parent / "oracle/sonic_compat_oracle.cpp"
        cls._temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls._temporary.name) / "sonic_compat_oracle"
        subprocess.run(
            [
                "g++", "-std=c++20", "-O2", str(source),
                f"-I{INCLUDE}", f"-I{DEPLOY / 'src'}",
                f"-I{CUDA / 'include'}", f"-I{TENSORRT / 'include'}",
                str(TRT_LIBRARY), f"-L{CUDA / 'lib64'}", f"-L{TENSORRT / 'lib'}",
                f"-Wl,-rpath,{CUDA / 'lib64'}:{TENSORRT / 'lib'}",
                "-lcudart", "-lnvinfer", "-lnvonnxparser", "-o", str(cls.binary),
            ],
            check=True,
        )
        output = subprocess.run(
            [str(cls.binary), str(MODEL)], check=True, text=True, capture_output=True
        ).stdout.splitlines()
        cls.oracle = {
            line.split(maxsplit=1)[0]: np.fromstring(line.split(maxsplit=1)[1], sep=" ")
            for line in output if line.startswith(("OBSERVATION ", "ACTION "))
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_observation_matches_cpp_oracle(self) -> None:
        builder = SonicObservationBuilder()
        token = -0.25 + 0.01 * np.arange(64)
        for frame in range(10):
            action = -0.5 + 0.04 * frame + 0.01 * np.arange(29)
            observation = builder.build(robot_state(frame), token, action)
        np.testing.assert_allclose(observation, self.oracle["OBSERVATION"], atol=1e-15)

    def test_onnx_output_matches_cpp_oracle(self) -> None:
        np.testing.assert_allclose(self.oracle["ACTION"], REFERENCE_ACTION, atol=1e-7)
        try:
            policy = OnnxPolicy(MODEL)
        except RuntimeError as error:
            self.skipTest(str(error))
        actual = policy(self.oracle["OBSERVATION"])
        np.testing.assert_allclose(actual, self.oracle["ACTION"], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
