import os
import argparse
import pickle

path = 'data/02_intermediate/marklinesEdges.p'
graph_object = pickle.load(open(path, "rb"))


def main(arguments) -> None:
    print('Hello.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")

    args = parser.parse_args()
    print(args)
    main(args)
