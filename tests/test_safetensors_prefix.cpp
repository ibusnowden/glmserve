#include "safetensors.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void write_file(const std::string& path, const std::string& s) {
    std::ofstream f(path, std::ios::binary);
    f << s;
}

static void write_shard(const std::string& path, const std::string& name, float value) {
    std::string header = "{\"" + name +
        "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    uint64_t n = header.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

int main() {
    char templ[] = "/tmp/glmserve_prefix_XXXXXX";
    char* d = mkdtemp(templ);
    if (!d) {
        std::perror("mkdtemp");
        return 1;
    }
    std::string dir(d);
    write_shard(dir + "/a.safetensors", "model.layers.0.foo.weight", 1.0f);
    write_shard(dir + "/b.safetensors", "model.layers.1.bar.weight", 2.0f);
    write_file(dir + "/model.safetensors.index.json",
               "{\"metadata\":{\"total_size\":8},\"weight_map\":{"
               "\"model.layers.0.foo.weight\":\"a.safetensors\","
               "\"model.layers.1.bar.weight\":\"b.safetensors\"}}");

    glmserve::SafeTensors st;
    st.load_prefixes(dir, {"model.layers.1."});
    if (st.has("model.layers.0.foo.weight")) {
        std::fprintf(stderr, "FAIL: prefix load included unrequested layer 0 tensor\n");
        return 1;
    }
    if (!st.has("model.layers.1.bar.weight")) {
        std::fprintf(stderr, "FAIL: prefix load missed requested layer 1 tensor\n");
        return 1;
    }
    float got = st.get("model.layers.1.bar.weight").as<float>()[0];
    if (got != 2.0f) {
        std::fprintf(stderr, "FAIL: tensor value %.3f != 2.0\n", got);
        return 1;
    }
    std::printf("test_safetensors_prefix: PASS\n");
    return 0;
}
