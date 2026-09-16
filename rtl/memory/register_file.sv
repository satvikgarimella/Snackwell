module register_file (
    input  logic        clk,
    input  logic        reset,

    input  logic [3:0]  rs1_addr,
    input  logic [3:0]  rs2_addr,

    output logic [31:0] rs1_data,
    output logic [31:0] rs2_data,

    input  logic        write_enable,
    input  logic [3:0]  rd_addr,
    input  logic [31:0] write_data
);

    logic [31:0] registers [0:15];

    integer i;

    always_ff @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 16; i = i + 1) begin
                registers[i] <= 32'b0;
            end
        end
        else if (write_enable) begin
            registers[rd_addr] <= write_data;
        end
    end

    always_comb begin
        rs1_data = registers[rs1_addr];
        rs2_data = registers[rs2_addr];
    end

endmodule