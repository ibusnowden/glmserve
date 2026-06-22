#include "tokenizer.hpp"
#include "common.hpp"
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace glmserve {

static bool exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}
static std::string read_text(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.good()) return "";
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// GPT-2 reversible byte<->unicode map: keeps text printable for BPE merges.
static void build_byte_maps(std::unordered_map<uint8_t, std::string>& enc,
                            std::unordered_map<std::string, uint8_t>& dec) {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    auto to_utf8 = [](int cp) {
        std::string s;
        if (cp < 0x80) s += static_cast<char>(cp);
        else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    };
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u = to_utf8(cs[i]);
        enc[static_cast<uint8_t>(bs[i])] = u;
        dec[u] = static_cast<uint8_t>(bs[i]);
    }
}

void Tokenizer::load(const std::string& model_dir, int64_t vocab_size) {
    vocab_size_ = vocab_size;
    std::string dir = model_dir;
    if (!dir.empty() && dir.back() != '/') dir += '/';

    std::string tok_json = dir + "tokenizer.json";
    if (exists(tok_json)) {
        try {
            load_bpe(tok_json);
            mode_ = Mode::BPE;
            GLM_INFO("tokenizer: BPE from %s (%zu tokens)", tok_json.c_str(), vocab_.size());
            return;
        } catch (const std::exception& e) {
            GLM_WARN("tokenizer.json present but unusable (%s); falling back to byte mode",
                     e.what());
        }
    }
    mode_ = Mode::Byte;
    GLM_INFO("tokenizer: byte-level fallback (vocab=%lld). For production fidelity "
             "wire a real GLM tokenizer.json or the python sidecar.",
             (long long)vocab_size_);
}

void Tokenizer::load_bpe(const std::string& tok_json) {
    auto root = json::parse(read_text(tok_json));
    GLM_CHECK(root && root->has("model"), "tokenizer.json missing model");
    auto model = root->at("model");
    GLM_CHECK(model->has("vocab"), "tokenizer.json model missing vocab");

    for (auto& [tok, id] : model->at("vocab")->obj) {
        int i = static_cast<int>(id->as_int());
        vocab_[tok] = i;
        id_to_token_[i] = tok;
    }
    auto merges = model->at("merges");
    if (merges && merges->is_array()) {
        int rank = 0;
        for (auto& m : merges->arr) {
            std::string s = m->as_string();
            auto sp = s.find(' ');
            if (sp == std::string::npos) continue;
            merge_rank_[{s.substr(0, sp), s.substr(sp + 1)}] = rank++;
        }
    }
    // added/special tokens
    if (root->has("added_tokens") && root->at("added_tokens")->is_array()) {
        for (auto& at : root->at("added_tokens")->arr) {
            std::string content = at->get_string("content");
            int id = static_cast<int>(at->get_int("id"));
            if (!content.empty()) {
                special_tokens_.push_back({content, id});
                vocab_[content] = id;
                id_to_token_[id] = content;
            }
        }
    }
    build_byte_maps(byte_encoder_, byte_decoder_);
    GLM_CHECK(!vocab_.empty(), "empty vocab");
}

std::vector<int> Tokenizer::bpe_encode(const std::string& text) const {
    // byte-level: map raw bytes through the GPT-2 byte encoder, then greedy BPE
    // over the resulting symbol sequence (no regex pre-tokenizer; best-effort).
    std::vector<std::string> symbols;
    for (unsigned char ch : text) {
        auto it = byte_encoder_.find(ch);
        symbols.push_back(it == byte_encoder_.end() ? std::string(1, static_cast<char>(ch))
                                                     : it->second);
    }
    if (symbols.empty()) return {};

    // iteratively merge the lowest-rank adjacent pair
    while (symbols.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_rank_.find({symbols[i], symbols[i + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second; best_i = i; found = true;
            }
        }
        if (!found) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }

    std::vector<int> ids;
    for (auto& s : symbols) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) ids.push_back(it->second);
        else for (char c : s) {  // unknown: emit per-byte ids if present
            auto bi = vocab_.find(std::string(1, c));
            if (bi != vocab_.end()) ids.push_back(bi->second);
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<int> ids;
    if (add_special && bos_id_ >= 0) ids.push_back(bos_id_);
    if (mode_ == Mode::Byte) {
        for (unsigned char c : text) ids.push_back(static_cast<int>(c));
    } else {
        auto e = bpe_encode(text);
        ids.insert(ids.end(), e.begin(), e.end());
    }
    return ids;
}

std::string Tokenizer::decode_token(int id) const {
    if (mode_ == Mode::Byte) {
        if (id >= 0 && id < 256) return std::string(1, static_cast<char>(id));
        return "";  // special / out-of-byte-range ids render empty
    }
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) return "";
    // reverse the byte encoder
    const std::string& tok = it->second;
    std::string out;
    size_t i = 0;
    while (i < tok.size()) {
        bool matched = false;
        for (size_t len = 3; len >= 1; --len) {
            if (i + len <= tok.size()) {
                auto bi = byte_decoder_.find(tok.substr(i, len));
                if (bi != byte_decoder_.end()) {
                    out += static_cast<char>(bi->second);
                    i += len; matched = true; break;
                }
            }
        }
        if (!matched) { out += tok[i]; ++i; }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode_token(id);
    return out;
}

bool Tokenizer::is_stop(int id) const { return id == eos_id_; }

// GLM chat template. The GLM-4.5/5.x format frames turns with role headers; in
// byte mode this is plain text, in BPE mode the special role tokens (if present
// in the vocab) are matched by encode().
std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& msgs,
                                           bool add_generation_prompt) const {
    std::string s = "[gMASK]<sop>";
    for (const auto& m : msgs) {
        s += "<|" + m.role + "|>\n" + m.content;
    }
    if (add_generation_prompt) s += "<|assistant|>\n";
    return s;
}

std::vector<int> Tokenizer::encode_chat(const std::vector<ChatMessage>& msgs) const {
    return encode(apply_chat_template(msgs, true), /*add_special=*/false);
}

}  // namespace glmserve
