package snackwell_pkg;

    localparam logic [5:0] OP_NOP       = 6'h00;
    localparam logic [5:0] OP_ADD       = 6'h01;
    localparam logic [5:0] OP_SUB       = 6'h02;
    localparam logic [5:0] OP_AND       = 6'h03;
    localparam logic [5:0] OP_OR        = 6'h04;
    localparam logic [5:0] OP_XOR       = 6'h05;
    localparam logic [5:0] OP_ADDI      = 6'h06;
    localparam logic [5:0] OP_TID       = 6'h07;

    localparam logic [5:0] OP_LD_SHARED = 6'h08;
    localparam logic [5:0] OP_ST_SHARED = 6'h09;

    localparam logic [5:0] OP_BEQ       = 6'h0A;
    localparam logic [5:0] OP_SSY       = 6'h0B;

    localparam logic [5:0] OP_EXIT      = 6'h3F;

endpackage
