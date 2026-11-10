"""Controlled serial/broadcast builds preserve outputs and reduce RAM requests."""

import unittest

from snackwell.assembler import assemble
from snackwell.cli.compare_global_loads import comparison_runner
from snackwell.kv_cache import Generation
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, checked_command, compile_testbench, tool_path
from tests.test_model import fixture


class GlobalLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/global_loads"
        cls.serial, _ = comparison_runner(serial=True)
        cls.broadcast, _ = comparison_runner(serial=False)

    def test_serial_baseline_handshake_masks_errors_reset_and_stores(self):
        output = self.output / "serial_unit"
        executable = compile_testbench(
            ROOT / "tb/tb_global_memory_unit.sv",
            output,
            defines=("SNACKWELL_SERIAL_GLOBAL_LOADS",),
        )
        log = checked_command([tool_path("vvp"), str(executable)], ROOT, output / "simulation.log")
        self.assertIn("broadcast=0", log)

    def test_repeated_loads_reduce_requests_distinct_loads_keep_cycles(self):
        program = assemble("""TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R1, [R12]
            LD_GLOBAL R2, [R1]
            ST_GLOBAL [R12 + 4160], R2
            EXIT
        """)
        memory = [0xDEADBEEF] * 16384
        memory[1024:1028] = [11, 22, 33, 44]
        with SimulatorSession(self.serial) as serial, SimulatorSession(self.broadcast) as broadcast:
            for addresses in ([4096] * 4, [4096, 4100, 4096, 4100], [4096, 4100, 4104, 4108]):
                shared = addresses + [0] * 1020
                for latency in (0, 2, 19):
                    a, b = [
                        s.run_global_words(program, shared, memory, 15, 1, latency, 10000)
                        for s in (serial, broadcast)
                    ]
                    self.assertEqual(a[:3], b[:3])
                    self.assertEqual(a[3]["reads"], 4)
                    self.assertEqual(b[3]["reads"], len(set(addresses)))
                    for result in (a, b):
                        self.assertEqual(result[0], shared)
                        expected = memory.copy()
                        expected[1040:1044] = [memory[addr // 4] for addr in addresses]
                        self.assertEqual(result[2], expected)
                        self.assertEqual(result[3]["writes"], 4)
                        self.assertEqual(result[3]["status"], 0)
                    if len(set(addresses)) == 4:
                        self.assertEqual(a[3]["cycles"], b[3]["cycles"])
                    else:
                        self.assertLess(b[3]["cycles"], a[3]["cycles"])

    def test_resident_generation_rejects_switching_build_configuration(self):
        self.assertNotEqual(self.serial.session_executable, self.broadcast.session_executable)
        with Generation(fixture(), "ab", memory="resident") as generation:
            generation.next(self.output / "first", self.serial)
            with self.assertRaisesRegex(ValueError, "cannot switch RTL builds"):
                generation.next(self.output / "second", self.broadcast)
            self.assertTrue(generation.closed)


if __name__ == "__main__":
    unittest.main()
