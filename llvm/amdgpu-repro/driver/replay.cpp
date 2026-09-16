// Replay a recorded gfx1250 kernel launch N times and count distinct output hashes.
//
// This driver deliberately understands nothing about what the kernel computes. It
// copies recorded bytes into device buffers, launches with a recorded kernarg blob,
// copies the output back, and hashes it. There is no reference implementation, no
// tolerance, and no arithmetic anywhere in this file.
//
// That is the whole point. The pass criterion is self-consistency:
//
//   N launches, byte-identical inputs re-uploaded before each one
//     -> 1 distinct output hash   : the kernel is deterministic
//     -> more than 1              : it returned different bytes from identical bytes
//
// A kernel that fails that test cannot be explained by a mis-written kernel or an
// unlucky reference, because nothing here is compared against anything except itself.
//
// Build:  hipcc -O2 -std=c++17 replay.cpp -o replay
// Usage:  ./replay --code K.hsaco --args <dir> [--runs 120] [--label name]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                            \
    do {                                                                           \
        hipError_t _e = (expr);                                                    \
        if (_e != hipSuccess) {                                                    \
            std::fflush(stdout);                                                   \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,     \
                         #expr, hipGetErrorString(_e));                            \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

static void fail(const std::string &msg) {
    // Flush first: stdout is buffered and stderr is not, so without this the
    // complaint appears above the lines saying what was being loaded when it
    // happened, which reads as though the context came after the failure.
    std::fflush(stdout);
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

// ---------------------------------------------------------------------------
// SHA-256. Present so the reported hash can be compared directly against hashes
// recorded by other tooling, rather than being a private checksum.
// ---------------------------------------------------------------------------

namespace sha256 {

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Ctx {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t len = 0;
    uint8_t buf[64];
    size_t n = 0;

    void block(const uint8_t *p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
                   (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t *p, size_t sz) {
        len += sz;
        while (sz) {
            size_t take = std::min(sz, size_t(64) - n);
            std::memcpy(buf + n, p, take);
            n += take; p += take; sz -= take;
            if (n == 64) { block(buf); n = 0; }
        }
    }

    std::string hex() {
        // Capture the message length before padding, since update() also counts
        // the padding bytes it is about to consume.
        uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (n != 56) update(&z, 1);
        uint8_t be[8];
        for (int i = 0; i < 8; ++i) be[i] = uint8_t(bits >> (56 - 8 * i));
        update(be, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

static std::string of(const std::vector<uint8_t> &v) {
    Ctx c;
    c.update(v.data(), v.size());
    return c.hex();
}

}  // namespace sha256

// ---------------------------------------------------------------------------
// The recorded launch
// ---------------------------------------------------------------------------

// AN ALLOCATION AND AN ARGUMENT ARE NOT THE SAME THING, AND ON THIS KERNEL THEY
// GENUINELY DIVERGE. The captured eight-wave launch hands the kernel arguments 22 and
// 23 as two windows onto ONE 12288-byte allocation, at byte offsets 0 and 2048. Giving
// each argument its own allocation would put the same bytes on the device at addresses
// standing in a different relation to one another than they did in the real launch --
// and if the kernel ever reads past the end of one window, the original reads its
// neighbour's bytes while the replay reads off the end of a smaller allocation. That is
// a divergence no output hash can show, in a driver whose whole claim is that a hash
// difference means something. So a Storage is allocated and uploaded, and a Pointer is
// an address INTO one.
struct Storage {
    int id = -1;
    size_t size = 0;
    std::string file;
    std::vector<uint8_t> host;  // recorded contents, re-uploaded before every launch
    void *dev = nullptr;
};

struct Pointer {
    int arg = -1;
    size_t kernarg_offset = 0;  // where this argument's device pointer goes
    int storage_id = -1;
    size_t byte_offset = 0;     // into that storage
    size_t extent = 0;          // the argument's own reachable span, used for the output
};

struct Launch {
    uint32_t grid[3] = {0, 0, 0};    // in workgroups
    uint32_t block[3] = {0, 0, 0};   // in work-items
    uint32_t lds = 0;
    std::string kernel;
    int output_buffer = -1;          // names an ARGUMENT, not a storage
    std::vector<uint8_t> kernarg;
    std::vector<Storage> storages;
    std::vector<Pointer> pointers;
};

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

// The manifest is a flat keyword-per-line format on purpose: the driver should
// have no dependency a reviewer has to install before reading it.
static Launch load(const std::string &dir) {
    Launch L;
    std::ifstream m(dir + "/manifest.txt");
    if (!m) fail("cannot read " + dir + "/manifest.txt");

    // Two capture formats are readable and they may not be mixed. The older one --
    // capture/capture_kernargs.py -- writes one `buffer` line per pointer argument and
    // refuses outright when an argument is a view into a shared allocation. The newer
    // one -- capture/capture_kernargs_by_storage.py -- writes a `storage` line per
    // allocation and a `pointer` line per argument, which is what the real launch
    // needed. A manifest carrying both would be two descriptions of the same kernarg
    // slots with nothing saying which wins, so it is refused rather than merged.
    bool saw_buffer = false, saw_storage_form = false;

    std::string line;
    while (std::getline(m, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        std::string key;
        s >> key;
        if (key == "grid")             s >> L.grid[0] >> L.grid[1] >> L.grid[2];
        else if (key == "block")       s >> L.block[0] >> L.block[1] >> L.block[2];
        else if (key == "lds")         s >> L.lds;
        else if (key == "kernel")      s >> L.kernel;
        else if (key == "output_buffer") s >> L.output_buffer;
        else if (key == "buffer") {
            // Old form: the argument index doubles as the storage id, the whole
            // allocation belongs to that one argument, and the offset into it is zero.
            saw_buffer = true;
            Storage b;
            Pointer p;
            std::string sha;
            s >> b.id >> p.kernarg_offset >> b.size >> b.file >> sha;
            b.host = read_file(dir + "/" + b.file);
            if (b.host.size() != b.size)
                fail("buffer " + b.file + " is " + std::to_string(b.host.size()) +
                     " bytes, manifest says " + std::to_string(b.size));
            if (!sha.empty() && sha != "-" && sha256::of(b.host) != sha)
                fail("buffer " + b.file + " does not match its recorded sha256");
            p.arg = b.id;
            p.storage_id = b.id;
            p.byte_offset = 0;
            p.extent = b.size;
            L.storages.push_back(std::move(b));
            L.pointers.push_back(p);
        } else if (key == "storage") {
            saw_storage_form = true;
            Storage b;
            std::string sha;
            s >> b.id >> b.size >> b.file >> sha;
            b.host = read_file(dir + "/" + b.file);
            if (b.host.size() != b.size)
                fail("storage " + b.file + " is " + std::to_string(b.host.size()) +
                     " bytes, manifest says " + std::to_string(b.size));
            if (!sha.empty() && sha != "-" && sha256::of(b.host) != sha)
                fail("storage " + b.file + " does not match its recorded sha256");
            L.storages.push_back(std::move(b));
        } else if (key == "pointer") {
            saw_storage_form = true;
            Pointer p;
            s >> p.arg >> p.kernarg_offset >> p.storage_id >> p.byte_offset >> p.extent;
            L.pointers.push_back(p);
        } else {
            fail("unknown manifest key '" + key + "'");
        }
    }

    if (saw_buffer && saw_storage_form)
        fail("this manifest mixes the one-buffer-per-argument form with the "
             "storage/pointer form. They describe the same kernarg slots and nothing "
             "says which wins; recapture rather than editing a manifest by hand.");

    L.kernarg = read_file(dir + "/kernarg.bin");
    if (L.kernel.empty()) fail("manifest names no kernel");
    if (L.output_buffer < 0) fail("manifest names no output_buffer");
    if (L.grid[0] == 0 || L.block[0] == 0) fail("manifest has no grid or block size");
    if (L.pointers.empty()) fail("manifest names no pointer arguments");

    // Everything below is checkable without a device, so it is checked without one.
    // --check-manifest runs the same loader, which is the point: a capture that would
    // be refused on the board is refused at a desk instead.
    std::map<int, const Storage *> by_id;
    for (const auto &b : L.storages) {
        if (by_id.count(b.id))
            fail("storage id " + std::to_string(b.id) + " appears twice");
        by_id[b.id] = &b;
    }
    bool have_output = false;
    for (const auto &p : L.pointers) {
        auto it = by_id.find(p.storage_id);
        if (it == by_id.end())
            fail("argument " + std::to_string(p.arg) + " names storage " +
                 std::to_string(p.storage_id) + ", which the manifest does not define");
        if (p.byte_offset + p.extent > it->second->size)
            fail("argument " + std::to_string(p.arg) + " reaches " +
                 std::to_string(p.byte_offset + p.extent) + " bytes into storage " +
                 std::to_string(it->second->id) + ", which is " +
                 std::to_string(it->second->size) + " bytes");
        if (p.kernarg_offset + sizeof(void *) > L.kernarg.size())
            fail("argument " + std::to_string(p.arg) + " puts its pointer at kernarg "
                 "offset " + std::to_string(p.kernarg_offset) + ", past the end of a " +
                 std::to_string(L.kernarg.size()) + "-byte blob");
        if (p.arg == L.output_buffer) {
            have_output = true;
            if (p.extent == 0) fail("the output argument has a zero-byte extent");
        }
    }
    if (!have_output) fail("output_buffer names no pointer argument in the manifest");
    return L;
}

int main(int argc, char **argv) {
    std::string code, args, label = "run";
    int runs = 120;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) fail("missing value for " + a); return std::string(argv[++i]); };
        if      (a == "--code")  code  = next();
        else if (a == "--args")  args  = next();
        else if (a == "--runs")  runs  = std::stoi(next());
        else if (a == "--label") label = next();
        // Parse the capture, print what it describes, and stop before touching the
        // device. A manifest mistake found here costs a second; found on the board it
        // costs a session on a machine a hundred and fifty tenants are sharing.
        else if (a == "--check-manifest") check_only = true;
        else fail("unknown argument " + a);
    }
    if (args.empty() || (code.empty() && !check_only))
        fail("usage: replay --code K.hsaco --args <dir> [--runs N] [--label name]\n"
             "       replay --args <dir> --check-manifest");

    Launch L = load(args);

    if (check_only) {
        std::printf("manifest    %s/manifest.txt\n", args.c_str());
        std::printf("kernel      %s\n", L.kernel.c_str());
        std::printf("grid        %u x %u x %u workgroups of %u x %u x %u, %u LDS bytes\n",
                    L.grid[0], L.grid[1], L.grid[2],
                    L.block[0], L.block[1], L.block[2], L.lds);
        std::printf("kernarg     %zu bytes\n", L.kernarg.size());
        size_t total = 0;
        for (const auto &b : L.storages) {
            std::printf("storage %3d %12zu bytes  %s\n", b.id, b.size, b.file.c_str());
            total += b.size;
        }
        std::printf("%zu allocation(s), %zu bytes total\n", L.storages.size(), total);
        for (const auto &pt : L.pointers)
            std::printf("argument %3d -> kernarg +%-4zu = storage %d + %zu, extent %zu%s\n",
                        pt.arg, pt.kernarg_offset, pt.storage_id, pt.byte_offset,
                        pt.extent, pt.arg == L.output_buffer ? "   [output]" : "");
        return 0;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device      %s\n", prop.gcnArchName);
    std::printf("code object %s\n", code.c_str());
    std::printf("kernel      %s\n", L.kernel.c_str());
    std::printf("grid        %u x %u x %u workgroups of %u x %u x %u, %u LDS bytes\n",
                L.grid[0], L.grid[1], L.grid[2],
                L.block[0], L.block[1], L.block[2], L.lds);

    hipModule_t mod;
    hipFunction_t fn;
    HIP_CHECK(hipModuleLoad(&mod, code.c_str()));
    HIP_CHECK(hipModuleGetFunction(&fn, mod, L.kernel.c_str()));

    // Ask the LOADED code object what geometry it will accept, and refuse rather than
    // launch if the manifest disagrees with it.
    //
    // This is not defensive padding. The two modules in this package carry the SAME
    // kernel name and the SAME 176-byte argument layout, but different work-group
    // limits: subject8 is .max_flat_workgroup_size 256 (eight waves at wave32) and
    // reference4 is 128 (four waves). A single captured manifest therefore cannot be
    // replayed into both, and the harness previously had nothing that would notice --
    // it launched whatever block size the manifest named into whatever object it was
    // handed. The best case was a runtime error with a confusing message; the case
    // worth preventing is a launch that proceeds and produces output bytes that are
    // wrong for a reason no output hash can show.
    //
    // The dynamic LDS request is checked for the same reason: this configuration asks
    // for 68408 bytes, which is far above the default limit, so a silent clamp or a
    // mismatched object would change what the kernel does without changing what the
    // driver prints.
    {
        int max_threads = 0, max_dyn_lds = 0;
        HIP_CHECK(hipFuncGetAttribute(&max_threads,
                                      HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
        HIP_CHECK(hipFuncGetAttribute(&max_dyn_lds,
                                      HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fn));
        const unsigned long long threads =
            (unsigned long long)L.block[0] * L.block[1] * L.block[2];
        std::printf("object accepts %d threads per workgroup, %d dynamic LDS bytes\n",
                    max_threads, max_dyn_lds);
        if (max_threads > 0 && threads > (unsigned long long)max_threads)
            fail("this manifest asks for " + std::to_string(threads) +
                 " threads per workgroup but " + code + " accepts at most " +
                 std::to_string(max_threads) +
                 ". These arguments were captured from a different module. Capture per"
                 " module rather than replaying one capture into both -- see"
                 " capture/README.md.");
        if (max_dyn_lds > 0 && L.lds > (unsigned)max_dyn_lds)
            fail("this manifest asks for " + std::to_string(L.lds) +
                 " dynamic LDS bytes but " + code + " accepts at most " +
                 std::to_string(max_dyn_lds) + ".");
    }

    // Allocate the recorded buffers and patch their device addresses into a copy of
    // the recorded kernarg blob. Scalar arguments are already inside that blob and
    // are never touched, so they cannot drift between the two variants.
    // load() has already checked that every pointer names a storage it fits inside and
    // lands in the kernarg blob, so what is left here is the part that needs a device.
    std::vector<uint8_t> kernarg = L.kernarg;
    std::map<int, Storage *> by_id;
    for (auto &b : L.storages) {
        HIP_CHECK(hipMalloc(&b.dev, b.size));
        by_id[b.id] = &b;
    }

    // Patch each argument's address in separately. Two arguments that named the same
    // storage get two addresses into the SAME allocation, the same distance apart as
    // they were on the device.
    const Pointer *out = nullptr;
    Storage *out_storage = nullptr;
    for (const auto &p : L.pointers) {
        Storage *b = by_id.at(p.storage_id);
        void *addr = static_cast<uint8_t *>(b->dev) + p.byte_offset;
        std::memcpy(kernarg.data() + p.kernarg_offset, &addr, sizeof(void *));
        if (p.arg == L.output_buffer) { out = &p; out_storage = b; }
    }
    // Read back the output ARGUMENT's own span, not its whole allocation: if it ever
    // shares one, the neighbouring bytes are not this kernel's result and hashing them
    // would report a difference the output does not have.
    void *out_addr = static_cast<uint8_t *>(out_storage->dev) + out->byte_offset;

    size_t kernarg_size = kernarg.size();
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernarg_size,
                      HIP_LAUNCH_PARAM_END};

    std::vector<uint8_t> host_out(out->extent);
    std::map<std::string, int> seen;   // ordered, so first-seen order is stable
    std::vector<std::string> order;

    for (int r = 0; r < runs; ++r) {
        // Re-upload EVERY allocation, including the one holding the output. A
        // corrupted result can therefore never feed the next launch, so run-to-run
        // differences cannot be explained by accumulated state.
        for (auto &b : L.storages)
            HIP_CHECK(hipMemcpy(b.dev, b.host.data(), b.size, hipMemcpyHostToDevice));

        HIP_CHECK(hipModuleLaunchKernel(fn, L.grid[0], L.grid[1], L.grid[2],
                                        L.block[0], L.block[1], L.block[2],
                                        L.lds, nullptr, nullptr, config));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(host_out.data(), out_addr, out->extent, hipMemcpyDeviceToHost));

        std::string h = sha256::of(host_out);
        if (seen.find(h) == seen.end()) order.push_back(h);
        seen[h]++;
    }

    std::printf("\n%s: %d runs, %zu distinct output hash(es)\n",
                label.c_str(), runs, order.size());
    for (const auto &h : order)
        std::printf("  %6d x  %s\n", seen[h], h.c_str());

    if (order.size() == 1) {
        std::printf("\nDETERMINISTIC: every run returned the same bytes.\n");
    } else {
        std::printf("\nNONDETERMINISTIC: identical input bytes produced %zu different\n"
                    "results across %d launches. Nothing in this driver interprets the\n"
                    "data, so this is a property of the compiled code alone.\n",
                    order.size(), runs);
    }

    for (auto &b : L.storages) HIP_CHECK(hipFree(b.dev));
    HIP_CHECK(hipModuleUnload(mod));
    return order.size() == 1 ? 0 : 2;
}
