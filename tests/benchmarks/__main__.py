"""One entry point for model, operator, and accuracy measurements."""

import argparse
from importlib import import_module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("model", "attention", "operators", "accuracy"),
                        help="accuracy runs Kunlun's full-reference stress tests")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="suite options; use SUITE --help for details")
    args = parser.parse_args(argv)
    return import_module(f".{args.suite}", __package__).main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
