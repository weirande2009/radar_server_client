import serial
import struct
import time
import numpy as np


def main():
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
    if not ser.is_open:
        print("fail to open USB")
        exit(1)
    else:
        print("open USB successfully")
    while True:
        data_list = [0xf1, 0, 0, 100.0, 100.0, 1, 1, 10, 0xf2]
        data = struct.pack("=BBBffBBBB", *data_list)
        result = ser.write(data)
        print("written data length: ", result)
        time.sleep(1)


if __name__ == "__main__":
    main()
    # data_list = [0xf1, 0, 0, 100.0, 100.0, 1, 1, 10, 0xf2]
    # data = struct.pack("=BBBffBBBB", *data_list)
    # print(data)
    # a = np.array([[[1, 2, 9], [3, 4, 10]], [[5, 6, 11], [7, 8, 12]]])
    # print(a.shape)
    # print(a[:, :, :2])

