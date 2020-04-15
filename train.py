from argparse import ArgumentParser
import logging
from src.train import Experiment


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the config file')
    return parser.parse_args()

def main():

    args = parse_args()
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
    experiment = Experiment(args.config_path)
    experiment.run()

if __name__ == '__main__':
    main()