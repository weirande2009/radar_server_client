import socket
import handler

MINIMAP = 0
MOVING = 1


class Client:
    def __init__(self, host="localhost", port=8888):
        self.sock = socket.socket()
        self.sock.connect((host, port))
        self.handler = handler.Handler()
        msg = self.sock.recv(1024).decode()
        print(msg)
        if msg == "welcome":
            return
        else:
            exit(-1)

    def send(self, data_list: list, cmd: int):
        message = self.handler.handle(data_list, cmd)
        self.sock.sendall(message)

    def close(self):
        self.sock.close()

    def receive(self):
        msg = self.sock.recv(1024).decode()
        return msg
