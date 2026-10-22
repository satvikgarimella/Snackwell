"""Check SIMD decode products against tensor execution and the integer FP32 oracle."""

import random
import unittest

from snackwell.kernels.inference import Layout, tiled_matmul
from snackwell.kernels.matvec import simd_matvec
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.simulator import compile_model_runner


class MatvecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/matvec"
        cls.executables = {
            backend: compile_model_runner(cls.output / backend, backend=backend)
            for backend in ("icarus", "auto")
        }

    def test_warp_partitions_transpose_accumulation_and_fp32_edges(self):
        rng = random.Random(571)
        edges = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x800000,
            0x3F800001,
            0xBF800001,
            0x33800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
        ]
        for warps in (1, 2, 3, 4):
            n, k = 4 * warps, 8
            for transpose in (False, True):
                for special in (False, True):
                    with self.subTest(warps=warps, transpose=transpose, special=special):

                        def word():
                            return rng.choice(edges) if special else bits(rng.uniform(-2, 2))

                        layout = Layout()
                        layout.allocate("a", k, [word() for _ in range(k)])
                        layout.allocate("b", n * k, [word() for _ in range(n * k)])
                        layout.allocate("out", n, [word() for _ in range(n)])
                        layout.allocate("bias", n, [word() for _ in range(n)])
                        options = dict(
                            transpose_b=transpose, accumulate=True, warps=warps, bias="bias"
                        )
                        outputs = []
                        for backend, executable in self.executables.items():
                            runtime = ModelRuntime(
                                self.output / f"{warps}_{transpose}_{special}_{backend}", executable
                            )
                            for source in (
                                simd_matvec(layout, n, k, **options),
                                tiled_matmul(layout, 1, n, k, **options),
                            ):
                                outputs.append(
                                    runtime.launch(
                                        "matvec",
                                        layout,
                                        source,
                                        "matmul",
                                        ("a", "b", "out", "bias"),
                                        "out",
                                        warp_mask=(1 << warps) - 1,
                                        m=1,
                                        n=n,
                                        k=k,
                                        transpose_b=transpose,
                                        accumulate=True,
                                        bias=True,
                                    )
                                )
                        self.assertTrue(all(result == outputs[0] for result in outputs))

    def test_grouped_rounding_and_bias_at_full_memory_capacity(self):
        # A fused MAC or a reassociated sum changes the first output.
        n, k = 28, 32
        a = [0xBF800000, 0x3F800001, 0, 0] * 8
        b = ([bits(1)] * n + [0x3F7FFFFE] * n + [0] * (2 * n)) * 8
        layout = Layout()
        layout.allocate("a", k, a)
        layout.allocate("b", n * k, b)
        layout.allocate("out", n)
        layout.allocate("bias", n, [bits(i / 8) for i in range(n)])
        runtime = ModelRuntime(self.output / "rounding", self.executables["auto"])
        outputs = []
        for source in (
            simd_matvec(layout, n, k, warps=4, bias="bias"),
            tiled_matmul(layout, 1, n, k, warps=4, bias="bias"),
        ):
            outputs.append(
                runtime.launch(
                    "matvec",
                    layout,
                    source,
                    "matmul",
                    ("a", "b", "bias"),
                    "out",
                    warp_mask=15,
                    m=1,
                    n=n,
                    k=k,
                    bias=True,
                )
            )
        self.assertEqual(outputs[0], [bits(i / 8) for i in range(n)])
        self.assertEqual(outputs[0], outputs[1])
        self.assertLess(runtime.reports[0]["cycles"], runtime.reports[1]["cycles"])
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))
