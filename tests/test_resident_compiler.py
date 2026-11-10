"""Execute both lowerings across loops, joins, aliases and register pressure."""

import unittest
from dataclasses import replace

from snackwell.assembler import assemble
from snackwell.direct_memory import DirectLaunch
from snackwell.kernels.inference import Layout
from snackwell.kv_cache import Generation
from snackwell.paths import ROOT
from snackwell.resident_memory import resident_memory
from snackwell.simulator import SimulatorSession, compile_model_runner
from tests.test_model import fixture


class ResidentCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/resident_compiler"
        cls.simulator = compile_model_runner(cls.output, "verilator", compact=True)

    def compare(self, name, layout, source, expected, *, output="out", warp_mask=1, latency=2):
        results = []
        for optimize in (False, True):
            with SimulatorSession(self.simulator) as session:
                device = resident_memory(session)
                launch = DirectLaunch(
                    device, layout, source, "test", {}, layout.regions[output], optimize=optimize
                )
                words = assemble(launch.source)
                result = launch.run(
                    words,
                    self.output / name / str(optimize),
                    active_masks=0xFFFF,
                    warp_mask=warp_mask,
                    latency=latency,
                )
                self.assertEqual(list(map(int, result[-1])), expected)
                results.append((result, launch, len(words)))
        self.assertEqual(results[0][0][2], results[1][0][2])
        return results

    def test_reused_pointer_base_changes_and_published_alias(self):
        layout = Layout()
        layout.allocate("x", 4, [11, 22, 33, 44])
        source = """XOR R15, R15, R15
            TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_GLOBAL R1, [R12]
            LD_GLOBAL R2, [R12]
            ADD R1, R1, R2
            ST_GLOBAL [R12], R1
            LD_GLOBAL R3, [R12]
            ADDI R3, R3, 7
            ST_GLOBAL [R12], R3
            EXIT
        """
        for latency in (0, 19):
            a, b = self.compare(
                "alias", layout, source, [29, 51, 73, 95], output="x", latency=latency
            )
            self.assertLess(b[0][3]["cycles"], a[0][3]["cycles"])
            self.assertLess(b[2], a[2])
        # Same register names after a base write must not reuse the old pointer.
        layout.allocate("out", 4)
        changed = (
            source[: source.index("            ST_GLOBAL")]
            + """
            ADDI R12, R12, 4
            LD_GLOBAL R3, [R12 - 4]
            ADD R3, R1, R3
            ST_GLOBAL [R12 + 12], R3
            EXIT
        """
        )
        self.compare("changed_base", layout, changed, [33, 66, 99, 132])

    def test_divergent_join_does_not_mistake_a_dynamic_address_for_a_literal(self):
        layout = Layout()
        out = layout.allocate("out", 16)
        a, b = layout.constant(10), layout.constant(20)
        self.assertEqual(b, a + 4)
        source = f"""XOR R15, R15, R15
            TID R12
            ADDI R3, R15, 1
            AND R2, R12, R3
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_GLOBAL R7, [R15 + {a}]
            SSY joined
            BEQ R2, R15, even
            ADDI R6, R15, 4
            BEQ R15, R15, joined
        even:
            XOR R6, R6, R6
        joined:
            LD_GLOBAL R1, [R6 + {a}]
            ADD R1, R1, R7
            ST_GLOBAL [R12 + {out}], R1
            EXIT
        """
        _, optimized = self.compare("join", layout, source, [20, 30] * 8, warp_mask=15)
        self.assertTrue(optimized[1].literals)

    def test_loop_backedge_widens_constant_addresses(self):
        layout = Layout()
        out = layout.allocate("out", 4)
        constants = [layout.constant(x) for x in (10, 20, 30, 40)]
        source = f"""XOR R15, R15, R15
            XOR R6, R6, R6
            ADDI R14, R15, 4
            LD_GLOBAL R7, [R15 + {constants[0]}]
        loop:
            LD_GLOBAL R1, [R6 + {constants[0]}]
            LD_GLOBAL R2, [R6 + {constants[0]}]
            ADD R1, R1, R2
            ADD R1, R1, R7
            ST_GLOBAL [R6 + {out}], R1
            ADDI R6, R6, 4
            ADDI R14, R14, -1
            BEQ R14, R15, done
            BEQ R15, R15, loop
        done:
            EXIT
        """
        self.compare("backedge", layout, source, [30, 50, 70, 90])

    def test_full_pointer_table_falls_back_without_overwriting_guards(self):
        layout = Layout()
        layout.allocate("unused", 1016)
        out = layout.allocate("out", 4)
        address = layout.constant(7)
        layout.constant(8)
        layout.constant(9)
        source = f"""XOR R15, R15, R15
            TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_GLOBAL R1, [R15 + {address}]
            ST_GLOBAL [R12 + {out}], R1
            EXIT
        """
        _, optimized = self.compare("full_table", layout, source, [7] * 4)
        self.assertFalse(optimized[1].literals)

    def test_all_registers_used_and_later_scratch_redefinition(self):
        layout = Layout()
        layout.allocate("x", 4, [1, 3, 5, 7])
        out = layout.allocate("out", 4)
        source = (
            "\n".join(f"XOR R{i}, R{i}, R{i}" for i in range(16))
            + f"""
            TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_GLOBAL R1, [R12]
            LD_GLOBAL R2, [R12]
            ADDI R14, R15, 9
            ADD R1, R1, R2
            ADD R1, R1, R14
            ST_GLOBAL [R12 + {out}], R1
            EXIT
        """
        )
        self.compare("registers", layout, source, [11, 15, 19, 23])

    def test_optimizer_generation_preserves_sampling_rebuild_and_cache_policy(self):
        model = fixture(context=4, width=8, layers=2, heads=2)
        for precision in ("fp32", "nvfp4"):
            with (
                Generation(
                    model, "abc", precision=precision, memory="resident", optimize_resident=False
                ) as a,
                Generation(model, "abc", precision=precision, memory="resident") as b,
            ):
                for i in range(3):
                    reports = [
                        g.next(
                            self.output / f"{precision}_{g.optimize_resident}_{i}",
                            self.simulator,
                            draw=0.4,
                        )
                        for g in (a, b)
                    ]
                    self.assertEqual(reports[0]["probabilities"], reports[1]["probabilities"])
                    self.assertEqual(reports[0]["selection"], reports[1]["selection"])
                    self.assertEqual(reports[0]["cache"]["phase"], reports[1]["cache"]["phase"])
                b.optimize_resident = False
                with self.assertRaisesRegex(ValueError, "KV cache"):
                    b.next(self.output / "switch", self.simulator)
                self.assertTrue(b.closed)

    def test_literal_corruption_cannot_publish_verified_output(self):
        from unittest.mock import patch

        original = SimulatorSession.run_resident_words
        with SimulatorSession(replace(self.simulator, compact=False)) as session:
            layout = Layout()
            layout.allocate("out", 4)
            address = layout.constant(10)
            source = f"""XOR R15, R15, R15
                TID R12
                ADD R12, R12, R12
                ADD R12, R12, R12
                LD_GLOBAL R1, [R15 + {address}]
                ST_GLOBAL [R12], R1
                EXIT
            """
            launch = DirectLaunch(
                resident_memory(session), layout, source, "test", {}, layout.regions["out"]
            )
            literal = next(iter(launch.literals.values())) // 4

            def corrupt(*args, **kwargs):
                shared, registers, values, metrics = original(*args, **kwargs)
                shared[literal] ^= 1
                return shared, registers, values, metrics

            with (
                patch.object(SimulatorSession, "run_resident_words", corrupt),
                self.assertRaisesRegex(ValueError, "shared guard"),
            ):
                launch.run(
                    assemble(launch.source),
                    self.output / "literal_corruption",
                    active_masks=15,
                    warp_mask=1,
                    latency=2,
                )
            self.assertIsNone(session.process)


if __name__ == "__main__":
    unittest.main()
