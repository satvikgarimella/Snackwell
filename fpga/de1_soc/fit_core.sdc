# Core-only 50 MHz target. This is not a measured clock frequency.
create_clock -name core_clk -period 20.000 [get_ports {clk}]
derive_clock_uncertainty
# The bus pins are virtual. No board I/O timing budget has been invented;
# external paths remain unconstrained and must be reviewed in the STA report.
