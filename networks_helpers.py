from argparse import ArgumentParser


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--mask-init", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="gumbel")
    parser.add_argument("--power", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1, help="Original Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.3,
        help="Factor to apply to LR when reducing",
    )
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience of lr scheduling in epochs"
    )
    parser.add_argument("--budget-penalty", type=float, default=1)

    return parser
