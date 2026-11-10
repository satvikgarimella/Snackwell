package snackwell_pkg;

    localparam logic [5:0] OP_NOP       = 6'h00;
    localparam logic [5:0] OP_ADD       = 6'h01;
    localparam logic [5:0] OP_SUB       = 6'h02;
    localparam logic [5:0] OP_AND       = 6'h03;
    localparam logic [5:0] OP_OR        = 6'h04;
    localparam logic [5:0] OP_XOR       = 6'h05;
    localparam logic [5:0] OP_ADDI      = 6'h06; // Rd = Rs1 + sign_extend(imm14)
    localparam logic [5:0] OP_TID       = 6'h07; // Rd[lane] = 4*warp_id + lane

    localparam logic [5:0] OP_LD_SHARED = 6'h08;
    localparam logic [5:0] OP_ST_SHARED = 6'h09;

    localparam logic [5:0] OP_BEQ       = 6'h0A;
    localparam logic [5:0] OP_SSY       = 6'h0B;

    localparam logic [5:0] OP_FMUL      = 6'h0C;
    localparam logic [5:0] OP_FADD      = 6'h0D;
    localparam logic [5:0] OP_FSUB      = 6'h0E;

    // TMUL Rd..Rd+3, Ra..Ra+3, Rb..Rb+3: one FP32 C=A*B tile.
    localparam logic [5:0] OP_TMUL      = 6'h0F;

    localparam logic [5:0] OP_SHR       = 6'h10; // logical Rs1 >> Rs2[4:0]
    localparam logic [5:0] OP_SLTU      = 6'h11; // unsigned less-than, integer 0/1
    localparam logic [5:0] OP_FLT       = 6'h12; // ordered binary32 less-than, integer 0/1

    // Reduced tensor-memory interface; FP32 prototype, not NVIDIA PTX encodings.
    localparam logic [5:0] OP_TALLOC    = 6'h13;
    localparam logic [5:0] OP_TFREE     = 6'h14;
    localparam logic [5:0] OP_TST       = 6'h15;
    localparam logic [5:0] OP_TLD       = 6'h16;
    localparam logic [5:0] OP_TMMA      = 6'h17;
    localparam logic [5:0] OP_TMMA_ACC  = 6'h18;
    localparam logic [5:0] OP_TCOMMIT   = 6'h19;
    localparam logic [5:0] OP_TWAIT     = 6'h1A;
    localparam logic [5:0] OP_TRELEASE  = 6'h1B;
    localparam logic [5:0] OP_TSTATUS   = 6'h1C;

    localparam logic [5:0] OP_LD_GLOBAL = 6'h1D;
    localparam logic [5:0] OP_ST_GLOBAL = 6'h1E;
    localparam logic [5:0] OP_GSTATUS   = 6'h1F;

    localparam logic [5:0] OP_CPASYNC   = 6'h20;
    localparam logic [5:0] OP_CPWAIT    = 6'h21;
    localparam logic [5:0] OP_CPRELEASE = 6'h22;
    localparam logic [5:0] OP_CPSTATUS  = 6'h23;

    localparam logic [5:0] OP_TMMA_NVFP4     = 6'h24;
    localparam logic [5:0] OP_TMMA_NVFP4_ACC = 6'h25;

    localparam logic [5:0] OP_EXIT      = 6'h3F;

    function automatic logic tensor_matrix_opcode(input logic [5:0] opcode);
        tensor_matrix_opcode = opcode == OP_TMMA || opcode == OP_TMMA_ACC ||
            opcode == OP_TMMA_NVFP4 || opcode == OP_TMMA_NVFP4_ACC;
    endfunction

    function automatic logic tensor_nvfp4_opcode(input logic [5:0] opcode);
        tensor_nvfp4_opcode = opcode == OP_TMMA_NVFP4 || opcode == OP_TMMA_NVFP4_ACC;
    endfunction

    // Both formats address four vectors; NVFP4 vectors contain sixteen packed
    // elements plus a padded scale word. FP32 vectors contain four words.
    function automatic logic [31:0] tensor_descriptor_last(
        input logic [31:0] descriptor, input logic nvfp4
    );
        tensor_descriptor_last = {20'd0, descriptor[11:0]} +
            3 * {20'd0, descriptor[23:12]} + (nvfp4 ? 11 : 15);
    endfunction

    function automatic logic tensor_descriptor_valid(
        input logic [31:0] descriptor, input logic nvfp4
    );
        tensor_descriptor_valid = descriptor[31:24] == 0 && descriptor[1:0] == 0 &&
            descriptor[13:12] == 0 && descriptor[23:12] >= (nvfp4 ? 12 : 16) &&
            tensor_descriptor_last(descriptor, nvfp4) < 4096;
    endfunction

endpackage
