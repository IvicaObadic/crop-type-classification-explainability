import argparse
from explainability_analysis.transformer_analysis import get_temporal_attn_weights


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_results_path',
        help='the root folder of the trained model')
    parser.add_argument(
        '--model_timestamp',
        help='the timestamp of the trained model')
    parser.add_argument(
        '--classes_to_exclude',
        default=None,
        help='occluded classes')
    parser.add_argument('--with_spectral_diff_as_input', action="store_true",
                        help='store the weights and gradients during test time')

    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    get_temporal_attn_weights(args.root_results_path,
                              args.model_timestamp,
                              args.classes_to_exclude,
                              args.with_spectral_diff_as_input)