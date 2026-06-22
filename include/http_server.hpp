// glmserve — minimal dependency-free HTTP/1.1 server (POSIX sockets).
//
// One thread per connection. Supports normal JSON responses and Server-Sent
// Events streaming (text/event-stream) for the OpenAI-compatible streaming API.
// Not a hardened public web server — it is the local serving front-end for a
// coding-agent harness on a trusted network.
#pragma once

#include <functional>
#include <map>
#include <string>

namespace glmserve {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::map<std::string, std::string> headers;
};

// Writes the response back to the client socket. Either call send() once, or
// start_sse() then sse()/sse_done() repeatedly for streaming.
class HttpResponder {
public:
    explicit HttpResponder(int fd) : fd_(fd) {}

    void send(int status, const std::string& content_type, const std::string& body);
    void send_json(int status, const std::string& json) {
        send(status, "application/json", json);
    }

    void start_sse();                 // emit SSE headers
    void sse(const std::string& data);// emit one `data: <...>\n\n` frame
    void sse_done();                  // emit `data: [DONE]\n\n`
    bool client_alive() const { return alive_; }

private:
    bool write_all(const char* buf, size_t n);
    int  fd_;
    bool sse_started_ = false;
    bool alive_ = true;
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponder&)>;

class HttpServer {
public:
    void route(const std::string& method, const std::string& path, HttpHandler h);
    // Blocks serving forever (until stop()). host e.g. "0.0.0.0".
    void listen_and_serve(const std::string& host, int port);
    void stop();

private:
    void handle_connection(int client_fd);
    std::map<std::string, HttpHandler> routes_;  // "METHOD path" -> handler
    int listen_fd_ = -1;
    volatile bool running_ = false;
};

}  // namespace glmserve
