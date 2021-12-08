import socket


def test_socket_port0():
    sock = socket.socket()
    x = sock.bind(('', 0))
    assert x is None
    ip, port = sock.getsockname()
    print('ip', ip)
    print('port', port)
    assert 0
