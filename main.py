import os
import argparse


def main(arguments) -> None:
    print('Hello.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")

    args = parser.parse_args()
    print(args)
    main(args)
