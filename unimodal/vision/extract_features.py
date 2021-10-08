import argparse

from utils import extractor

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Extract Features')

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default='cfg.json', metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--verbose', type=bool, default=True,
                        help='Whether to print to stdout.')
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")


def main():
    """Main function."""

    args = parser.parse_args()

    extractor(args)

if __name__ == '__main__':
    main()