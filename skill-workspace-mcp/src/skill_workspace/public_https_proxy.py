"""Per-session HTTPS CONNECT proxy for an otherwise offline local MCP.

The untrusted MCP joins only an internal Docker network. This trusted sidecar
also joins Docker's outbound bridge, resolves and pins a public destination,
and never accepts plaintext HTTP or private-address CONNECT targets.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import select
import socket
import socketserver


BENCHMARK_PROXY = ipaddress.ip_network("198.18.0.0/15")
HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
FORBIDDEN_SUFFIXES = (".localhost", ".local", ".internal", ".test", ".invalid")
MAX_HEADER = 16 * 1024
MAX_TRANSFER = 128 * 1024 * 1024


def valid_host(host: str) -> bool:
    if not isinstance(host, str) or not host or len(host) > 253 or host != host.lower():
        return False
    if host in {"localhost", "host.docker.internal"} or host.endswith(FORBIDDEN_SUFFIXES):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        return len(labels) >= 2 and all(HOST_LABEL.fullmatch(label) for label in labels)
    return False


def parse_policy(value: str) -> tuple[str, ...]:
    hosts = json.loads(value)
    if not isinstance(hosts, list) or not 1 <= len(hosts) <= 16:
        raise ValueError("public HTTPS policy requires 1–16 hosts")
    if hosts == ["*"]:
        return ("*",)
    if "*" in hosts or any(not valid_host(host) for host in hosts):
        raise ValueError("public HTTPS policy contains an invalid host")
    return tuple(dict.fromkeys(hosts))


def admitted_addresses(host: str):
    if not valid_host(host):
        raise ValueError("HTTPS target is not a public DNS name")
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    public = [item for item in addresses if ipaddress.ip_address(item[4][0]).is_global]
    if public:
        return public
    # Docker Desktop can map a public hostname to its managed 198.18/15
    # tunnel. The original hostname is still constrained above and the client
    # performs end-to-end TLS verification after CONNECT.
    managed = [item for item in addresses
               if ipaddress.ip_address(item[4][0]) in BENCHMARK_PROXY]
    if managed:
        return managed
    raise ValueError("HTTPS target resolved only to private or reserved addresses")


def connect_public(host: str):
    last_error = None
    for family, socktype, protocol, _, address in admitted_addresses(host):
        upstream = socket.socket(family, socktype, protocol)
        upstream.settimeout(15)
        try:
            upstream.connect(address)
            upstream.settimeout(None)
            return upstream
        except OSError as error:
            last_error = error
            upstream.close()
    raise OSError("public HTTPS connection failed") from last_error


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        if not self.server.slots.acquire(blocking=False):
            self.request.sendall(b"HTTP/1.1 503 Busy\r\nConnection: close\r\n\r\n")
            return
        upstream = None
        try:
            header = bytearray()
            while not header.endswith(b"\r\n\r\n") and len(header) <= MAX_HEADER:
                byte = self.request.recv(1)
                if not byte:
                    return
                header.extend(byte)
            if len(header) > MAX_HEADER:
                raise ValueError("proxy request headers exceeded limit")
            request_line = header.split(b"\r\n", 1)[0].decode("ascii")
            method, authority, version = request_line.split(" ")
            if method != "CONNECT" or version != "HTTP/1.1" or authority.count(":") != 1:
                raise ValueError("only HTTPS CONNECT to port 443 is supported")
            host, port = authority.rsplit(":", 1)
            host = host.lower()
            if (port != "443" or not valid_host(host)
                    or ("*" not in self.server.hosts and host not in self.server.hosts)):
                raise ValueError("HTTPS target is outside the approved host policy")
            upstream = connect_public(host)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = (self.request, upstream)
            transferred = [0, 0]
            while True:
                readable, _, _ = select.select(sockets, [], [], 300)
                if not readable:
                    break
                for incoming in readable:
                    block = incoming.recv(64 * 1024)
                    if not block:
                        return
                    index = 0 if incoming is self.request else 1
                    transferred[index] += len(block)
                    if transferred[index] > MAX_TRANSFER:
                        raise ValueError("HTTPS tunnel exceeded transfer limit")
                    sockets[1 - index].sendall(block)
        except (OSError, UnicodeError, ValueError):
            if upstream is None:
                try:
                    self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                except OSError:
                    pass
        finally:
            if upstream is not None:
                upstream.close()
            self.server.slots.release()


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, hosts):
        import threading
        self.hosts = hosts
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(("0.0.0.0", 3128), Handler)


if __name__ == "__main__":
    with Server(parse_policy(os.environ["EXOHARNESS_EGRESS_HOSTS"])) as server:
        server.serve_forever()
