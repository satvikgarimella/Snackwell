// Untimed Verilator host. File and word transports share the same RTL clock loop.
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

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint32_t frame_magic = 0x534e5731;

uint32_t read_word() {
    std::array<unsigned char, 4> bytes;
    if (!std::cin.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Incomplete word transport request");
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
           uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}

void send_word(uint32_t value) {
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24)};
    std::cout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

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
    if (argc != 2 || (std::string(argv[1]) != "+SESSION" && std::string(argv[1]) != "+WORDS")) {
        std::cerr << "ERROR This executable requires +SESSION or +WORDS\n";
        return 1;
    }
    const bool word_transport = std::string(argv[1]) == "+WORDS";
#ifdef _WIN32
    if (word_transport) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
    }
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
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
        while (std::cin.peek() != std::char_traits<char>::eof()) {
            uint32_t active, warps, entry, limit;
            std::array<uint32_t, 1024> program, memory;
            if (word_transport) {
                if (read_word() != frame_magic) throw std::runtime_error("Invalid word frame");
                active = read_word(); warps = read_word();
                entry = read_word(); limit = read_word();
                for (auto& word : program) word = read_word();
                for (auto& word : memory) word = read_word();
            } else {
                if (!std::getline(std::cin, directory) || directory.empty() ||
                    !std::getline(std::cin, options))
                    throw std::runtime_error("Incomplete session request");
                std::array<std::string, 5> fields;
                std::string extra;
                std::istringstream parsed(options);
                for (auto& field : fields)
                    if (!(parsed >> field)) throw std::runtime_error("Missing launch option");
                if (parsed >> extra) throw std::runtime_error("Extra launch option");
                active = number(fields[0], 16); warps = number(fields[1], 16);
                entry = number(fields[2], 10); limit = number(fields[3], 10);
                if (number(fields[4], 10) != 0)
                    throw std::runtime_error("Traces require the SV testbench");
                program = read_image(directory + "/program.hex");
                memory = read_image(directory + "/input.hex");
            }
            if (active > 0xffff || warps == 0 || warps > 15 ||
                limit == 0 || limit > 1000000)
                throw std::runtime_error("Invalid launch options");

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
            if (word_transport) {
                send_word(frame_magic);
                send_word(cycles);
                for (int i = 0; i < 1024; ++i) send_word(top.snackwell_sim_top->read_word(i));
                for (int i = 0; i < 256; ++i) send_word(top.snackwell_sim_top->read_register(i));
                std::cout.flush();
                if (!std::cout) throw std::runtime_error("Cannot write word response");
            } else {
                write_image(directory + "/output.hex", 1024,
                            [&](int i) { return top.snackwell_sim_top->read_word(i); });
                write_image(directory + "/registers.hex", 256,
                            [&](int i) { return top.snackwell_sim_top->read_register(i); });
                std::cout << "PASS kernel completed in " << cycles << " cycles\n" << std::flush;
            }
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
