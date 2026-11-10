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
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint32_t frame_magic = 0x534e5731;
constexpr uint32_t global_magic = 0x534e4731;
constexpr uint32_t resident_magic = 0x534e5231;
constexpr uint32_t resident_words = 262144;

struct Range { uint32_t start, count; };
uint32_t read_word();

Range read_range() {
    const Range range{read_word(), read_word()};
    if (!range.count || range.start >= resident_words || range.count > resident_words - range.start)
        throw std::runtime_error("Invalid resident range");
    return range;
}

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
    if (argc != 2 || (std::string(argv[1]) != "+SESSION" && std::string(argv[1]) != "+WORDS" &&
                      std::string(argv[1]) != "+GLOBAL" && std::string(argv[1]) != "+RESIDENT")) {
        std::cerr << "ERROR This executable requires +SESSION, +WORDS, +GLOBAL or +RESIDENT\n";
        return 1;
    }
    const bool resident = std::string(argv[1]) == "+RESIDENT";
    const bool global_transport = resident || std::string(argv[1]) == "+GLOBAL";
    const bool word_transport = global_transport || std::string(argv[1]) == "+WORDS";
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
    top.global_enable = global_transport;
    top.resident_enable = resident;
    top.global_latency = 0;
    if (resident)
        for (unsigned i = 0; i < resident_words; ++i)
            top.snackwell_sim_top->load_global_word(i, 0xdeadbeef);
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
            std::array<uint32_t, 16384> external;
            std::vector<Range> readback, writable;
            std::vector<std::pair<uint32_t, uint32_t>> uploads;
            std::vector<uint32_t> before;
            const uint32_t magic = resident ? resident_magic : global_transport ? global_magic : frame_magic;
            if (word_transport) {
                if (read_word() != magic)
                    throw std::runtime_error("Invalid word frame");
                active = read_word(); warps = read_word();
                entry = read_word(); limit = read_word();
                if (global_transport) {
                    top.global_latency = read_word();
                    if (top.global_latency > 1000) throw std::runtime_error("Invalid memory latency");
                }
                for (auto& word : program) word = read_word();
                for (auto& word : memory) word = read_word();
                if (resident) {
                    const auto patches = read_word();
                    if (patches > resident_words) throw std::runtime_error("Too many resident patches");
                    for (unsigned p = 0; p < patches; ++p) {
                        const auto range = read_range();
                        if (uploads.size() + range.count > resident_words)
                            throw std::runtime_error("Too much resident upload data");
                        for (unsigned j = 0; j < range.count; ++j)
                            uploads.emplace_back(range.start + j, read_word());
                    }
                    for (auto* ranges : {&readback, &writable}) {
                        const auto count = read_word();
                        if (count > resident_words) throw std::runtime_error("Too many resident ranges");
                        for (unsigned i = 0; i < count; ++i) ranges->push_back(read_range());
                    }
                } else if (global_transport) for (auto& word : external) word = read_word();
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
            if (resident) {
                for (const auto& [index, word] : uploads)
                    top.snackwell_sim_top->load_global_word(index, word);
                before.reserve(resident_words);
                for (unsigned i = 0; i < resident_words; ++i)
                    before.push_back(top.snackwell_sim_top->read_global_word(i));
            } else if (global_transport)
                for (int i = 0; i < 16384; ++i)
                    top.snackwell_sim_top->load_global_word(i, external[i]);
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
            if (resident) {
                std::vector<bool> allowed(resident_words, false);
                for (const auto& range : writable)
                    for (unsigned j = 0; j < range.count; ++j) allowed[range.start+j] = true;
                for (unsigned i = 0; i < resident_words; ++i)
                    if (!allowed[i] && before[i] != top.snackwell_sim_top->read_global_word(i))
                        throw std::runtime_error("Resident memory guard changed at word " + std::to_string(i));
            }
            if (word_transport) {
                send_word(magic);
                send_word(cycles);
                for (int i = 0; i < 1024; ++i) send_word(top.snackwell_sim_top->read_word(i));
                for (int i = 0; i < 256; ++i) send_word(top.snackwell_sim_top->read_register(i));
                if (global_transport) {
                    if (resident) {
                        for (const auto& range : readback)
                            for (unsigned j = 0; j < range.count; ++j)
                                send_word(top.snackwell_sim_top->read_global_word(range.start+j));
                    } else {
                        for (int i = 0; i < 16384; ++i)
                            send_word(top.snackwell_sim_top->read_global_word(i));
                    }
                    for (uint32_t metric : {uint32_t(top.global_reads), uint32_t(top.global_writes),
                         uint32_t(top.global_responses), uint32_t(top.global_stalls),
                         uint32_t(top.global_errors), uint32_t(top.copy_errors),
                         uint32_t(top.copy_reads), uint32_t(top.copy_writes),
                         uint32_t(top.overlap_cycles), uint32_t(top.narrow_macs), uint32_t(top.wide_multiplies)})
                        send_word(metric);
                }
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
