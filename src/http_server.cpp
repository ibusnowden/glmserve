#include "http_server.hpp"
#include "common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <thread>

namespace glmserve {

bool HttpResponder::write_all(const char* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::write(fd_, buf + off, n - off);
        if (w <= 0) { alive_ = false; return false; }
        off += static_cast<size_t>(w);
    }
    return true;
}

void HttpResponder::send(int status, const std::string& ctype, const std::string& body) {
    const char* reason = status == 200 ? "OK" : status == 404 ? "Not Found"
                         : status == 400 ? "Bad Request" : status == 500
                         ? "Internal Server Error" : "OK";
    std::ostringstream h;
    h << "HTTP/1.1 " << status << " " << reason << "\r\n"
      << "Content-Type: " << ctype << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Connection: close\r\n\r\n";
    std::string head = h.str();
    if (write_all(head.data(), head.size())) write_all(body.data(), body.size());
}

void HttpResponder::start_sse() {
    sse_started_ = true;
    const char* head =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    write_all(head, std::strlen(head));
}

void HttpResponder::sse(const std::string& data) {
    std::string frame = "data: " + data + "\n\n";
    write_all(frame.data(), frame.size());
}

void HttpResponder::sse_done() {
    const char* d = "data: [DONE]\n\n";
    write_all(d, std::strlen(d));
}

// ---------------------------------------------------------------------------

void HttpServer::route(const std::string& method, const std::string& path, HttpHandler h) {
    routes_[method + " " + path] = std::move(h);
}

static bool read_request(int fd, HttpRequest& req) {
    std::string buf;
    char tmp[8192];
    // read until we have headers (\r\n\r\n)
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t r = ::read(fd, tmp, sizeof(tmp));
        if (r <= 0) return false;
        buf.append(tmp, static_cast<size_t>(r));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20) && header_end == std::string::npos) return false;
    }
    std::string head = buf.substr(0, header_end);
    std::string rest = buf.substr(header_end + 4);

    std::istringstream hs(head);
    std::string line;
    std::getline(hs, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream ls(line);
        std::string target;
        ls >> req.method >> target;
        auto qm = target.find('?');
        if (qm == std::string::npos) req.path = target;
        else { req.path = target.substr(0, qm); req.query = target.substr(qm + 1); }
    }
    size_t content_length = 0;
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = line.substr(0, c);
        std::string v = line.substr(c + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
        for (auto& ch : k) ch = static_cast<char>(::tolower(ch));
        req.headers[k] = v;
        if (k == "content-length") content_length = std::stoul(v);
    }

    // read remaining body
    req.body = rest;
    while (req.body.size() < content_length) {
        ssize_t r = ::read(fd, tmp, sizeof(tmp));
        if (r <= 0) break;
        req.body.append(tmp, static_cast<size_t>(r));
    }
    return true;
}

void HttpServer::handle_connection(int client_fd) {
    int one = 1;
    ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    HttpRequest req;
    if (read_request(client_fd, req)) {
        HttpResponder res(client_fd);
        if (req.method == "OPTIONS") {  // CORS preflight
            res.send(200, "text/plain", "");
        } else {
            auto it = routes_.find(req.method + " " + req.path);
            if (it != routes_.end()) {
                try {
                    it->second(req, res);
                } catch (const std::exception& e) {
                    GLM_ERROR("handler exception: %s", e.what());
                    res.send_json(500, std::string("{\"error\":{\"message\":\"") +
                                       e.what() + "\"}}");
                }
            } else {
                res.send_json(404, "{\"error\":{\"message\":\"not found\"}}");
            }
        }
    }
    ::close(client_fd);
}

void HttpServer::listen_and_serve(const std::string& host, int port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    GLM_CHECK(listen_fd_ >= 0, "socket() failed");
    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = (host == "0.0.0.0") ? INADDR_ANY : inet_addr(host.c_str());

    GLM_CHECK(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
              "bind(%s:%d) failed: %s", host.c_str(), port, std::strerror(errno));
    GLM_CHECK(::listen(listen_fd_, 64) == 0, "listen() failed");

    running_ = true;
    GLM_INFO("HTTP server listening on http://%s:%d", host.c_str(), port);

    while (running_) {
        sockaddr_in cli{};
        socklen_t cl = sizeof(cli);
        int cfd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&cli), &cl);
        if (cfd < 0) { if (running_) continue; else break; }
        std::thread(&HttpServer::handle_connection, this, cfd).detach();
    }
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

void HttpServer::stop() {
    running_ = false;
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
}

}  // namespace glmserve
