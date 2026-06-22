// glmserve — minimal dependency-free JSON parser.
// Supports objects, arrays, strings (with \uXXXX), numbers, bool, null.
// Enough for config.json and safetensors headers; not a general-purpose lib.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace glmserve {
namespace json {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value {
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<ValuePtr> arr;
    std::map<std::string, ValuePtr> obj;

    bool is_null()   const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array()  const { return type == Type::Array; }

    // Convenience accessors with defaults.
    bool has(const std::string& k) const {
        return type == Type::Object && obj.count(k) > 0;
    }
    ValuePtr at(const std::string& k) const {
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : it->second;
    }
    int64_t as_int(int64_t def = 0) const {
        return type == Type::Number ? static_cast<int64_t>(num) : def;
    }
    double as_double(double def = 0.0) const {
        return type == Type::Number ? num : def;
    }
    bool as_bool(bool def = false) const {
        return type == Type::Bool ? b : def;
    }
    std::string as_string(const std::string& def = "") const {
        return type == Type::String ? str : def;
    }
    int64_t get_int(const std::string& k, int64_t def = 0) const {
        auto v = at(k);
        return v ? v->as_int(def) : def;
    }
    double get_double(const std::string& k, double def = 0.0) const {
        auto v = at(k);
        return v ? v->as_double(def) : def;
    }
    bool get_bool(const std::string& k, bool def = false) const {
        auto v = at(k);
        return v ? v->as_bool(def) : def;
    }
    std::string get_string(const std::string& k, const std::string& def = "") const {
        auto v = at(k);
        return v ? v->as_string(def) : def;
    }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    ValuePtr parse() {
        skip_ws();
        ValuePtr v = parse_value();
        skip_ws();
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(i_) + ": " + msg);
    }
    char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
    char get()  { return i_ < s_.size() ? s_[i_++] : '\0'; }
    void skip_ws() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i_;
            else break;
        }
    }
    void expect(char c) { if (get() != c) fail(std::string("expected '") + c + "'"); }

    ValuePtr parse_value() {
        skip_ws();
        char c = peek();
        switch (c) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return parse_string_value();
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default:  return parse_number();
        }
    }

    ValuePtr parse_object() {
        auto v = std::make_shared<Value>();
        v->type = Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') { get(); return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v->obj[key] = parse_value();
            skip_ws();
            char c = get();
            if (c == ',') continue;
            if (c == '}') break;
            fail("expected ',' or '}' in object");
        }
        return v;
    }

    ValuePtr parse_array() {
        auto v = std::make_shared<Value>();
        v->type = Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v->arr.push_back(parse_value());
            skip_ws();
            char c = get();
            if (c == ',') continue;
            if (c == ']') break;
            fail("expected ',' or ']' in array");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = get();
            if (c == '\0') fail("unterminated string");
            if (c == '"') break;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        // surrogate pair
                        if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                            s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            i_ += 2;
                            uint32_t lo = parse_hex4();
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    uint32_t parse_hex4() {
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = get();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) {
            out += static_cast<char>(cp);
        } else if (cp <= 0x7FF) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    ValuePtr parse_string_value() {
        auto v = std::make_shared<Value>();
        v->type = Type::String;
        v->str = parse_string();
        return v;
    }

    ValuePtr parse_bool() {
        auto v = std::make_shared<Value>();
        v->type = Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v->b = true;  i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v->b = false; i_ += 5; }
        else fail("bad literal");
        return v;
    }

    ValuePtr parse_null() {
        auto v = std::make_shared<Value>();
        v->type = Type::Null;
        if (s_.compare(i_, 4, "null") == 0) i_ += 4;
        else fail("bad literal");
        return v;
    }

    ValuePtr parse_number() {
        auto v = std::make_shared<Value>();
        v->type = Type::Number;
        size_t start = i_;
        if (peek() == '-' || peek() == '+') get();
        while (true) {
            char c = peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') get();
            else break;
        }
        if (i_ == start) fail("invalid number");
        v->num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

inline ValuePtr parse(const std::string& s) { return Parser(s).parse(); }

}  // namespace json
}  // namespace glmserve
