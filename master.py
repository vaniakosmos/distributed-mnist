from typing import List

import grpc
import tensorflow as tf

import train_pb2
import train_pb2_grpc
from model import FakeNet, Net


def generate_tensors(net: Net):
    for name, data in net.weights_iter():
        tensor = train_pb2.Tensor(name=name, data=data)
        yield tensor


def train(net: Net, stubs: List[train_pb2_grpc.TrainerStub]):
    new_tensors = []
    for stub in stubs:
        responses = stub.Train(generate_tensors(net))
        new_tensors.append(responses)

    for tensors in zip(*new_tensors):
        print(tensors)
        # name = tensors[0].name
        # tensor = tf.add_n([t.data for t in tensors])
        # tensor = tf.reduce_mean(tensor)
        # net.set_weights(name, tensor)


def main():
    workers = ['localhost:8001', 'localhost:8002']
    channels = [
        grpc.insecure_channel(uri)
        for uri in workers
    ]
    stubs = [
        train_pb2_grpc.TrainerStub(channel)
        for channel in channels
    ]
    net = FakeNet()
    train(net, stubs)

    for channel in channels:
        channel.close()


if __name__ == '__main__':
    main()
