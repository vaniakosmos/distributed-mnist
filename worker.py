import datetime
import time
from concurrent import futures

import grpc
import sys

import train_pb2
import train_pb2_grpc
from model import Net, FakeNet


class Server(train_pb2_grpc.TrainerServicer):
    def __init__(self, net: Net):
        self.net = net
        self.epoch_steps = 2

    def Train(self, request_iterator, context):
        print('inside ::Train')
        for tensor in request_iterator:
            print(tensor)
            self.net.set_weights(tensor.name, tensor.data)

        for i in range(self.epoch_steps):
            print('training', i)
            self.net.train()

        for (name, data) in self.net.weights_iter():
            print('sending back:', name)
            tensor = train_pb2.Tensor(name=name, data=data)
            yield tensor


def main():
    if len(sys.argv) == 2:
        port = sys.argv[1]
    else:
        port = '8001'
    net = FakeNet()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    train_pb2_grpc.add_TrainerServicer_to_server(Server(net), server)
    server.add_insecure_port('localhost:' + port)
    server.start()
    print('started')
    try:
        while True:
            time.sleep(datetime.timedelta(days=1).seconds)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()
