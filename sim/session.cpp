// Untimed Verilator host. stdin carries the same requests as the SV testbench.
// Traced and one-shot execution continue to use tb_snackwell_kernel.sv.
#include "Vsnackwell_sim_top.h"
#include "Vsnackwell_sim_top_snackwell_sim_top.h"
#include "verilated.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
uint32_t number(const std::string& text, int base) {
    uint32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Invalid unsigned word: " + text);
    return value;
}

std::array<uint32_t, 1024> read_image(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open image: " + path);
    std::array<uint32_t, 1024> image{};
    size_t index = 0;
    std::string line, word;
    while (std::getline(input, line)) {
        const auto comment = line.find("//");
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream words(line);
        while (words >> word) {
            if (index == image.size() || word.size() != 8)
                throw std::runtime_error("Expected 1024 eight-digit words: " + path);
            image[index++] = number(word, 16);
        }
    }
    if (input.bad() || index != image.size())
        throw std::runtime_error("Incomplete image: " + path);
    return image;
}

template <typename ReadWord>
void write_image(const std::string& path, int size, ReadWord read) {
    std::ofstream output(path);
    output << std::hex << std::setfill('0');
    for (int index = 0; index < size; ++index)
        output << std::setw(8) << read(index) << '\n';
    output.close();
    if (!output) throw std::runtime_error("Cannot write image: " + path);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "+SESSION") {
        std::cerr << "ERROR This executable requires +SESSION\n";
        return 1;
    }
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Vsnackwell_sim_top top{&context};
    // Each tick ends with settled outputs on the falling edge. Input changes
    // outside this loop are evaluated separately before the next rising edge.
    const auto tick = [&]() {
        top.clk = 1;
        top.eval();
        top.clk = 0;
        top.eval();
        if (context.gotFinish()) throw std::runtime_error("RTL ended before completion");
    };
    try {
        std::string directory, options;
        while (std::getline(std::cin, directory)) {
            if (directory.empty() || !std::getline(std::cin, options))
                throw std::runtime_error("Incomplete session request");
            std::array<std::string, 5> fields;
            std::string extra;
            std::istringstream parsed(options);
            for (auto& field : fields)
                if (!(parsed >> field)) throw std::runtime_error("Missing launch option");
            if (parsed >> extra) throw std::runtime_error("Extra launch option");
            const auto active = number(fields[0], 16), warps = number(fields[1], 16);
            const auto entry = number(fields[2], 10), limit = number(fields[3], 10);
            if (active > 0xffff || warps == 0 || warps > 15 ||
                limit == 0 || limit > 1000000 || number(fields[4], 10) != 0)
                throw std::runtime_error("Invalid launch options; traces require the SV testbench");
            const auto program = read_image(directory + "/program.hex");
            const auto memory = read_image(directory + "/input.hex");

            top.reset = 1;
            top.run_enable = top.launch_enable = 0;
            top.clk = 0;
            top.eval();
            for (int i = 0; i < 3; ++i) tick();
            top.reset = 0;
            for (int i = 0; i < 1024; ++i) {
                top.snackwell_sim_top->load_word(true, i, program[i]);
                top.snackwell_sim_top->load_word(false, i, memory[i]);
            }
            top.eval();
            for (unsigned warp = 0; warp < 4; ++warp) {
                if (!(warps & (1u << warp))) continue;
                tick();
                top.launch_enable = 1;
                top.launch_warp_id = warp;
                top.launch_pc = entry;
                top.launch_active_mask = (active >> (4 * warp)) & 15;
                top.eval();
                tick();
                top.launch_enable = 0;
                top.eval();
            }
            top.run_enable = 1;
            top.eval();
            unsigned cycles = 0;
            while ((top.halted_mask & warps) != warps && cycles < limit) {
                tick();
                ++cycles;
            }
            if ((top.halted_mask & warps) != warps)
                throw std::runtime_error("Kernel timeout after " + std::to_string(cycles) + " cycles");
            if (top.pending) throw std::runtime_error("Kernel halted before all work completed");
            top.run_enable = 0;
            write_image(directory + "/output.hex", 1024,
                        [&](int i) { return top.snackwell_sim_top->read_word(i); });
            write_image(directory + "/registers.hex", 256,
                        [&](int i) { return top.snackwell_sim_top->read_register(i); });
            std::cout << "PASS kernel completed in " << cycles << " cycles\n" << std::flush;
        }
        if (std::cin.bad()) throw std::runtime_error("Cannot read session request");
    } catch (const std::exception& error) {
        top.final();
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
    top.final();
    return 0;
}
