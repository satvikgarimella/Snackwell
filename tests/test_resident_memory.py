"""Persistent memory must preserve execution, ownership and numerical checks."""

import copy
import gc
import unittest
from unittest.mock import patch

from snackwell.assembler import assemble
from snackwell.kv_cache import Generation
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import SENTINEL, bits
from snackwell.paths import ROOT
from snackwell.resident_memory import CAPACITY, DeviceWord, resident_memory
from snackwell.simulator import SimulatorSession, compile_model_runner
from tests.test_model import fixture


class ResidentMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/resident_memory"
        cls.simulator = compile_model_runner(cls.output, "verilator", compact=True)

    def test_transport_preserves_untouched_data_and_checks_all_guards(self):
        load = assemble("XOR R0, R0, R0\nLD_GLOBAL R1, [R0]\nEXIT")
        store = assemble("XOR R0, R0, R0\nADDI R1, R0, 19\nST_GLOBAL [R0 + 4], R1\nEXIT")
        with SimulatorSession(self.simulator) as session:
            args = (load, [SENTINEL] * 1024, [(0, [123])], [(0, 2)], [])
            _, registers, after, metrics = session.run_resident_words(*args)
            self.assertEqual(after, [123, SENTINEL])
            self.assertEqual(registers[1], 123)
            self.assertEqual(metrics["host_upload_bytes"], 4)
            _, _, after, metrics = session.run_resident_words(load, args[1], [], [(0, 2)], [])
            self.assertEqual(after, [123, SENTINEL])
            self.assertEqual(metrics["host_upload_bytes"], 0)
            with self.assertRaisesRegex(RuntimeError, "memory guard"):
                session.run_resident_words(store, args[1], [], [], [])
            self.assertIsNone(session.process)
            # A fresh process has fresh memory, not silently restored old data.
            _, _, after, _ = session.run_resident_words(load, args[1], [], [(0, 2)], [])
            self.assertEqual(after, [SENTINEL] * 2)

    def test_rtl_outputs_feed_consumers_without_host_reupload(self):
        with SimulatorSession(self.simulator) as session:
            runtime = ModelRuntime(self.output / "chain", session, memory="resident")
            device = runtime.device
            a = device.constant("a", [bits(x / 8) for x in range(32)])
            b = device.constant("b", [bits(x / 16) for x in range(64)])
            result = runtime.matmul("matrix", a, b, 4, 8, 8)
            self.assertTrue(all(isinstance(w, DeviceWord) for w in result))
            before = device.uploaded_constant_bytes
            again = runtime.matmul("matrix", a, b, 4, 8, 8)
            self.assertEqual(result, again)
            self.assertEqual(device.uploaded_constant_bytes, before)
            self.assertEqual(runtime.reports[-1]["memory"]["device_copy_words"], 0)
            self.assertLess(
                runtime.reports[-1]["memory"]["total"]["host_upload_bytes"], 4 * len(a + b)
            )
            runtime.rows("activation", "gelu", result, 4, 8)
            corrupted = list(result)
            corrupted[0] = DeviceWord(int(result[0]) ^ 1, result[0].allocation, result[0].offset)
            with self.assertRaisesRegex(ValueError, "shadow"):
                runtime.rows("tampered", "gelu", corrupted, 4, 8)

    def test_prefill_decode_rebuild_sampling_match_global_in_both_precisions(self):
        model = fixture(context=4, width=8, heads=2, layers=2)
        for precision in ("fp32", "nvfp4"):
            with (
                Generation(
                    model, "abc", heal=False, precision=precision, memory="global"
                ) as baseline,
                Generation(
                    model, "abc", heal=False, precision=precision, memory="resident"
                ) as resident,
            ):
                for step in range(4):
                    results = [
                        g.next(
                            self.output / f"{precision}_{g.memory}_{step}",
                            self.simulator,
                            draw=0.4,
                            temperature=0.7,
                        )
                        for g in (baseline, resident)
                    ]
                    self.assertEqual(results[0]["probabilities"], results[1]["probabilities"])
                    self.assertEqual(results[0]["selection"], results[1]["selection"])
                    report = results[1]
                    self.assertEqual(report["resident"]["constant_upload_bytes"] == 0, step > 0)
                    self.assertEqual(report["launches"], report["resident"]["rtl_launches"])
                    self.assertEqual(
                        report["resident"]["responses"],
                        report["resident"]["reads"] + report["resident"]["writes"],
                    )
                    self.assertTrue(
                        all(
                            isinstance(w, DeviceWord)
                            for k, v in resident.cache.layers
                            for w in k + v
                        )
                    )
                self.assertEqual(report["resident"]["device_copy_words"], 0)
                self.assertLess(
                    report["resident"]["host_upload_bytes"], results[0]["launches"] * 65536
                )
                session = resident.session
            self.assertIsNone(session.process)

    def test_allocations_reclaim_and_reject_foreign_stale_or_oversized_handles(self):
        with SimulatorSession(self.simulator) as first, SimulatorSession(self.simulator) as second:
            a, b = resident_memory(first), resident_memory(second)
            initial = list(a.free)
            block = a.allocate(511)
            start = block.start
            values = a.words(block, [SENTINEL] * 511)
            with self.assertRaisesRegex(ValueError, "foreign"):
                b.address(values[0])
            with self.assertRaises(MemoryError):
                a.allocate(CAPACITY)
            del block, values
            gc.collect()
            self.assertEqual(a.free, initial)
            block = a.allocate(511)
            self.assertEqual(block.start, start)
            stale = a.words(block, [SENTINEL] * 511)
            first.close()
            with self.assertRaisesRegex(ValueError, "stale"):
                a.address(stale[0])

    def test_large_scatter_and_memory_latency_do_not_change_results(self):
        data = [bits((i % 17 - 8) / 16) for i in range(512)]
        results = []
        for latency in (0, 9):
            with SimulatorSession(self.simulator) as session:
                runtime = ModelRuntime(
                    self.output / f"large_{latency}",
                    session,
                    memory="resident-copy",
                    memory_latency=latency,
                )
                result = runtime.rows("gelu", "gelu", data, 4, 128)
                results.append(list(map(int, result)))
                self.assertGreaterEqual(runtime.reports[0]["memory"]["copy_launches"], 2)
        self.assertEqual(*results)

    def test_invalid_ranges_and_timeouts_fail_without_reusing_device_state(self):
        program = assemble("XOR R0, R0, R0\nLD_GLOBAL R1, [R0]\nEXIT")
        with SimulatorSession(self.simulator) as session:
            for uploads, reads in (([(CAPACITY, [1])], []), ([], [(-1, 1)]), ([], [(0, 0)])):
                with self.assertRaises(ValueError):
                    session.run_resident_words(program, [SENTINEL] * 1024, uploads, reads, [])
            device = resident_memory(session)
            handles = device.constant("weight", [1, 2, 3])
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                session.run_resident_words(program, [SENTINEL] * 1024, [], [], [], max_cycles=1)
            self.assertIsNone(session.process)
            with self.assertRaisesRegex(ValueError, "stale"):
                device.address(handles[0])

    def test_model_change_and_cancellation_close_the_session(self):
        model = fixture()
        with Generation(model, "a", heal=False, memory="resident") as generation:
            generation.next(self.output / "binding", self.simulator)
            old = generation.session
            changed = copy.deepcopy(model)
            changed["weights"]["lm_bias"][0] += 0.5
            generation.model = changed
            with self.assertRaisesRegex(ValueError, "different model"):
                generation.next(self.output / "wrong_binding", self.simulator)
            self.assertIsNone(old.process)
        with Generation(model, "a", heal=False, memory="resident") as generation:

            def cancel(event):
                if event["type"] == "kernel_complete":
                    raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                generation.next(self.output / "cancelled", self.simulator, cancel)
            self.assertIsNone(generation.session)
            self.assertFalse((self.output / "cancelled/result.json").exists())

    def test_corrupt_readback_cannot_become_a_verified_cache(self):
        with Generation(fixture(), "a", heal=False, memory="resident") as generation:
            original = SimulatorSession.run_resident_words

            def corrupt(session, *args, **kwargs):
                shared, registers, after, metrics = original(session, *args, **kwargs)
                if after:
                    after[0] ^= 1
                return shared, registers, after, metrics

            with (
                patch.object(SimulatorSession, "run_resident_words", corrupt),
                self.assertRaises(ValueError),
            ):
                generation.next(self.output / "corrupt", self.simulator)
            self.assertFalse(generation.cache.ids)
            self.assertIsNone(generation.session)
            self.assertFalse((self.output / "corrupt/result.json").exists())


if __name__ == "__main__":
    unittest.main()
