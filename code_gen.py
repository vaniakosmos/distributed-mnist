from grpc_tools import protoc


def main():
    protoc.main((
        '',
        '-I./protos',
        '--python_out=.',
        '--grpc_python_out=.',
        './protos/train.proto',
    ))


if __name__ == '__main__':
    main()
