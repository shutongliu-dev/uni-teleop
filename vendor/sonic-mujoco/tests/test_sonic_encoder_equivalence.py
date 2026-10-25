import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_sonic_encoder import robot_state, teleop_command

from sonic_mujoco.controllers.sonic import OnnxEncoder, SonicEncoderObservationBuilder

REFERENCE_ROOT = Path(os.environ.get("SONIC_REFERENCE_ROOT", "/nonexistent"))
DEPLOY = REFERENCE_ROOT / "gear_sonic_deploy"
MODEL = DEPLOY / "policy/release/model_encoder.onnx"
INCLUDE = DEPLOY / "src/g1/g1_deploy_onnx_ref/include"
CUDA = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))
TENSORRT = Path(os.environ.get("TENSORRT_ROOT", "/nonexistent"))
TRT_LIBRARY = DEPLOY / "build/src/TRTInference/libTRTInference.a"


class SonicEncoderEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (MODEL, CUDA, TENSORRT, TRT_LIBRARY)
        if not all(path.exists() for path in required) or shutil.which("g++") is None:
            raise unittest.SkipTest("SONIC encoder C++ reference is not available")

        source = Path(__file__).parent / "oracle/sonic_encoder_oracle.cpp"
        cls._temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls._temporary.name) / "sonic_encoder_oracle"
        subprocess.run(
            [
                "g++",
                "-std=c++20",
                "-O2",
                str(source),
                f"-I{INCLUDE}",
                f"-I{DEPLOY / 'src'}",
                f"-I{CUDA / 'include'}",
                f"-I{TENSORRT / 'include'}",
                str(TRT_LIBRARY),
                f"-L{CUDA / 'lib64'}",
                f"-L{TENSORRT / 'lib'}",
                f"-Wl,-rpath,{CUDA / 'lib64'}:{TENSORRT / 'lib'}",
                "-lcudart",
                "-lnvinfer",
                "-lnvonnxparser",
                "-o",
                str(cls.binary),
            ],
            check=True,
        )
        output = subprocess.run(
            [str(cls.binary), str(MODEL)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        cls.oracle = {
            line.split(maxsplit=1)[0]: np.fromstring(line.split(maxsplit=1)[1], sep=" ")
            for line in output
            if line.startswith(("OBSERVATION ", "TOKEN "))
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_observation_matches_cpp_oracle(self) -> None:
        builder = SonicEncoderObservationBuilder()
        builder.update(teleop_command(0, 10))
        actual = builder.build(robot_state())
        np.testing.assert_allclose(actual, self.oracle["OBSERVATION"], atol=1e-15)

    def test_onnx_output_matches_cpp_tensorrt(self) -> None:
        try:
            encoder = OnnxEncoder(MODEL)
        except RuntimeError as error:
            self.skipTest(str(error))
        actual = encoder(self.oracle["OBSERVATION"])
        np.testing.assert_allclose(actual, self.oracle["TOKEN"], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
