import serial
import struct
import crc_table

MINIMAP = 0
MOVING = 1


class Communicator:
    head = 0xf1
    tail = 0xf2

    def __init__(self):
        self.communicator = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)

    def send(self, data_list: list, cmd: int) -> None:
        sent_msg = "".encode()
        if cmd == MINIMAP:
            sent_msg = self.generate_minimap_sent_bytes(data_list)
        elif cmd == MOVING:
            sent_msg = self.generate_moving_sent_bytes(data_list)
        self.communicator.write(sent_msg)
        print("sent message: ", sent_msg)

    def generate_minimap_sent_bytes(self, data_list: list) -> bytes:
        tmp_list = [self.head, MINIMAP, data_list[0]]
        for i in range(len(data_list[1])):
            tmp_list.append(data_list[1][i][0])
            tmp_list.append(data_list[1][i][1])
        tmp_list.append(data_list[2])
        for i in range(len(data_list[3])):
            tmp_list.append(data_list[3][i][0])
            tmp_list.append(data_list[3][i][1])
        return self.generate_bytes(tmp_list)

    def generate_moving_sent_bytes(self, data_list: list) -> bytes:
        tmp_list = [self.head, MINIMAP, data_list[0], data_list[1]]
        return self.generate_bytes(tmp_list)

    def generate_bytes(self, tmp_list: list) -> bytes:
        struct_format_str = "="
        for i in range(len(tmp_list)):
            struct_format_str += "B"
        tmp_bytes = struct.pack(struct_format_str, *tmp_list)
        data = self.compute_crc8(tmp_bytes)
        print("Command: ", tmp_list[1])
        print("bytes: ", tmp_bytes)
        print("crc8: ", data)
        remain_bytes = struct.pack("BB", *[data, self.tail])
        tmp_bytes += remain_bytes
        return tmp_bytes

    def compute_crc8(self, data: bytes) -> bytes:
        ucCRC8 = crc_table.CRC8_INIT
        for i in range(len(data)):
            ucIndex = ucCRC8 ^ data[i]
            ucCRC8 = crc_table.CRC8_Table[ucIndex]
        return ucCRC8

if __name__ == "__main__":
    data_list = [4, [[1, 3], [3, 4], [5, 6], [7, 8], [0, 0], [0, 0], [0, 0], [0, 0]],
                 4, [[1, 2], [3, 4], [5, 6], [7, 8], [0, 0], [0, 0], [0, 0], [0, 0]]]
    com = Communicator()
    com.send(data_list, MINIMAP)



