import logging
from typing import List

import grpc

import train_pb2
import train_pb2_grpc
from model import Net


logger = logging.getLogger(__name__)


def generate_tensors(net: Net):
    for name, dtype, data in net.weights_iter():
        tensor = train_pb2.Tensor(name=name, data=data, dtype=dtype)
        logger.debug('get tensor: %s (%s)', name, dtype)
        yield tensor


def train(net: Net, stubs: List[train_pb2_grpc.TrainerStub]):
    new_tensors = []
    for stub in stubs:
        logger.debug('send tensors to worker %s', stub)
        responses = stub.Train(generate_tensors(net))
        new_tensors.append(responses)

    for tensors in zip(*new_tensors):
        logger.debug([t.name for t in tensors])
        name = tensors[0].name
        dtype = tensors[0].dtype
        net.set_weights(name, dtype, [t.data for t in tensors])


def main():
    workers = ['localhost:8001']
    channels = [
        grpc.insecure_channel(uri)
        for uri in workers
    ]
    stubs = [
        train_pb2_grpc.TrainerStub(channel)
        for channel in channels
    ]
    net = Net()
    for i in range(1000):
        train(net, stubs)

    for channel in channels:
        channel.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
