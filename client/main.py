import client

MINIMAP = 0
MOVING = 1

if __name__ == "__main__":
    host = "localhost"
    port = 8888
    my_client = client.Client(host, port)
    msg = my_client.receive()
    print(msg)
    if msg == "welcome":
        while True:
            a = input("press any key to send message...")
            my_client.send([100, 100], MOVING)
