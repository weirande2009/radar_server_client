from protobufs import Minimap_pb2
from protobufs import Moving_pb2


class ProtobufOut:
    def __init__(self, data_list: list):
        self.data = None
        self.data_list = data_list

    def generate(self):
        pass

    def get_data(self) -> bytes:
        return self.data


class MinimapOut(ProtobufOut):
    def __init__(self, data_list: list):
        super().__init__(data_list)
        self.generate()

    def generate(self):
        proto = Minimap_pb2.Minimap()
        proto.friendNumber = self.data_list[0]
        for i in range(len(self.data_list[1])):
            position = proto.fPositions.add()
            position.x = self.data_list[1][i][0]
            position.y = self.data_list[1][i][1]
        proto.enemyNumber = self.data_list[2]
        for i in range(len(self.data_list[3])):
            position = proto.ePositions.add()
            position.x = self.data_list[3][i][0]
            position.y = self.data_list[3][i][1]
        self.data = proto.SerializeToString()


class MovingOut(ProtobufOut):
    def __init__(self, data_list):
        super().__init__(data_list)
        self.generate()

    def generate(self):
        proto = Moving_pb2.Moving()
        proto.x = self.data_list[0]
        proto.y = self.data_list[1]
        self.data = proto.SerializeToString()

