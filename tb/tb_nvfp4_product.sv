`timescale 1ns/1ps
module tb_nvfp4_product;
    logic [3:0] a, b;
    logic [7:0] scale_a, scale_b;
    wire [31:0] product;
    logic [31:0] expected;
    integer file_handle, parsed, count;
    reg [4095:0] path;
    nvfp4_product dut (.*);
    initial begin
        if (!$value$plusargs("NVFP4_VECTORS=%s", path)) $fatal(1, "Missing NVFP4 vectors");
        file_handle = $fopen(path, "r");
        if (!file_handle) $fatal(1, "Cannot open NVFP4 vectors");
        count = 0;
        while (!$feof(file_handle)) begin
            parsed = $fscanf(file_handle, "%h %h %h %h %h\n", a, scale_a, b, scale_b, expected);
            if (parsed == 5) begin
                #1;
                if (product !== expected)
                    $fatal(1, "NVFP4 %h * %h, scales %h %h: %h expected %h", a, b, scale_a, scale_b, product, expected);
                count = count + 1;
            end else if (!$feof(file_handle)) $fatal(1, "Malformed vector");
        end
        if (count != 37888) $fatal(1, "Incomplete vector suite: %0d", count);
        $fclose(file_handle);
        $display("PASS NVFP4 product: %0d vectors, all element/scale encodings and signed zero", count);
        $finish;
    end
endmodule
