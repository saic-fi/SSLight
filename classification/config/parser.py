"""Argument parser functions."""

import argparse
import sys

from config.defaults import get_cfg


def parse_args():
    """
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Image-based Self Supervised Learning Evaluation."
    )
    parser.add_argument(
        "--job_name",
        help="The name of the job when using MLP",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--node_rank",
        help="The id of the node (it's important when using multiple nodes)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="config/experiments/BYOL.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Inherit parameters from args.
    if hasattr(args, "node_rank"):
        cfg.NODE_RANK = args.node_rank
    
    return cfg
