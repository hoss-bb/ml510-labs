"""Prepare, verify and train a separate model candidate."""
import argparse
from .helpers.workflow import run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-alpha", type=float, default=10)
    run_job(parser.parse_args().ridge_alpha)


if __name__ == "__main__":
    main()
