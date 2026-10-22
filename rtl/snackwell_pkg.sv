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

    localparam logic [5:0] OP_EXIT      = 6'h3F;

endpackage
