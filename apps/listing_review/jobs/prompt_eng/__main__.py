"""Compare prompt candidates and export the selected instructions."""
import argparse
from .helpers.workflow import run_job


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_job()


if __name__ == "__main__":
    main()
