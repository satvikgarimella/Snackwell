module instruction_memory #(
    parameter integer WORDS = 1024,
    parameter INIT_FILE = ""
) (
    input  logic [31:0] pc,
    output logic [31:0] instruction
);
    import snackwell_pkg::*;
    localparam logic [31:0] EXIT_INSTRUCTION = {OP_EXIT, 26'd0};
    logic [31:0] memory [0:WORDS-1];
    string program_file;
    integer file_handle;

    initial begin
        if (WORDS < 4) $fatal(1, "Instruction memory needs at least four words");
        program_file = INIT_FILE;
        // Simulation program loading; no runtime instruction-write bus yet.
        if ($value$plusargs("PROGRAM=%s", program_file)) begin end
        for (integer i = 0; i < WORDS; i = i + 1)
            memory[i] = program_file == "" ? 32'd0 : EXIT_INSTRUCTION;
        if (program_file != "") begin
            file_handle = $fopen(program_file, "r");
            if (file_handle == 0) $fatal(1, "Cannot open program: %s", program_file);
            $fclose(file_handle);
            $readmemh(program_file, memory);
        end else begin
            // Preserve the original smoke-test program when no image is given.
            memory[0] = {OP_ADD, 4'd3, 4'd1, 4'd2, 14'd0};
            memory[1] = {OP_SUB, 4'd4, 4'd1, 4'd2, 14'd0};
            memory[2] = {OP_XOR, 4'd5, 4'd1, 4'd2, 14'd0};
            memory[3] = EXIT_INSTRUCTION;
        end
    end

    // Never wrap a high or unaligned PC back into another instruction.
    // Invalid fetches halt via EXIT; this version has no instruction trap.
    always @(*) begin
        instruction = EXIT_INSTRUCTION;
        if (pc[1:0] == 2'b0 && pc[31:2] < WORDS)
            instruction = memory[pc[31:2]];
    end
endmodule
