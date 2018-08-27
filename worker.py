import datetime
import logging
import sys
import time
from concurrent import futures

import grpc

import train_pb2
import train_pb2_grpc
from model import Net


logger = logging.getLogger(__name__)


class Server(train_pb2_grpc.TrainerServicer):
    def __init__(self, net: Net):
        self.net = net
        self.epoch_steps = 100

    def Train(self, request_iterator, context):
        logger.debug('inside ::Train')
        for tensor in request_iterator:
            logging.debug(tensor.name)
            self.net.set_weights(tensor.name, tensor.dtype, tensor.data)

        loss = 'undefined'
        for i in range(self.epoch_steps):
            logger.debug('training %d', i)
            loss = self.net.train()
        logger.info('loss: %s', loss)

        for name, dtype, data in self.net.weights_iter():
            logger.debug('sending back: %s', name)
            tensor = train_pb2.Tensor(name=name, data=data, dtype=dtype)
            yield tensor


def main():
    if len(sys.argv) == 2:
        port = sys.argv[1]
    else:
        port = '8001'
    net = Net()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    train_pb2_grpc.add_TrainerServicer_to_server(Server(net), server)
    server.add_insecure_port('localhost:' + port)
    server.start()
    logger.info('started')
    try:
        while True:
            time.sleep(datetime.timedelta(days=1).seconds)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
