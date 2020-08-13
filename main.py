from appearance_bias.api import regress
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help="The emotional label to predict. If none provided, will evaluate all."
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        help='Path to folders of images to test on.'
    )
    parser.add_argument(
        '--cross_validate',
        '-v',
        action='store_true',
        help='Whether or not to cross validate the model.'
    )
    parser.add_argument(
        '--image_dirs',
        type=str,
        nargs='+',
        default='',
        required=True,
        help='Path to folders of labeled images.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    regress(**vars(FLAGS))