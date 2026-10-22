# Proposed 50 MHz core clock; meeting it has NOT been verified.
# External I/O delays belong to the eventual board transport.
create_clock -name core_clk -period 20.000 [get_ports {clk}]
