// glmserve — tokenizer + chat template.
//
// Two backends:
//   * Byte-level fallback (default): deterministic, vocab-free, maps each UTF-8
//     byte to an id. Always correct for round-trips; used for bring-up, tests,
//     and any model whose real tokenizer isn't wired yet. (Spec §9.2 V0 path.)
//   * tokenizer.json BPE (best-effort): loads vocab + merges and runs GPT-2
//     byte-level BPE. Enable when a real GLM tokenizer.json is present.
//
// For production fidelity, tools/export_tokenizer.py can also bridge to the HF
// tokenizer out-of-process; see docs/serving.md.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace glmserve {

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant" | "tool"
    std::string content;
};

class Tokenizer {
public:
    enum class Mode { Byte, BPE };

    // Loads from a model directory. Falls back to byte mode if no usable
    // tokenizer.json is found. base_vocab is used to size the byte vocab.
    void load(const std::string& model_dir, int64_t vocab_size);

    std::vector<int> encode(const std::string& text, bool add_special = false) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_token(int id) const;  // single-token piece (for streaming)

    // GLM-style chat template -> a single prompt string, then encode().
    std::string apply_chat_template(const std::vector<ChatMessage>& msgs,
                                    bool add_generation_prompt = true) const;
    std::vector<int> encode_chat(const std::vector<ChatMessage>& msgs) const;

    int  eos_id() const { return eos_id_; }
    int  bos_id() const { return bos_id_; }
    bool is_stop(int id) const;
    Mode mode() const { return mode_; }
    int64_t vocab_size() const { return vocab_size_; }

    void set_special(int bos, int eos) { bos_id_ = bos; eos_id_ = eos; }

private:
    void load_bpe(const std::string& tok_json);
    std::vector<int> bpe_encode(const std::string& text) const;

    Mode mode_ = Mode::Byte;
    int64_t vocab_size_ = 256;
    int bos_id_ = 1, eos_id_ = 2;

    // BPE state
    std::unordered_map<std::string, int> vocab_;        // token string -> id
    std::unordered_map<int, std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_rank_;
    std::unordered_map<uint8_t, std::string> byte_encoder_;  // GPT-2 byte map
    std::unordered_map<std::string, uint8_t> byte_decoder_;
    std::vector<std::pair<std::string, int>> special_tokens_;  // (text, id)
};

}  // namespace glmserve
