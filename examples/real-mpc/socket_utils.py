# socket_utils.py
import struct
import pickle


def send(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def recv(sock):
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack("!I", raw_len)[0]
    return pickle.loads(_recv_exact(sock, length))


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf
