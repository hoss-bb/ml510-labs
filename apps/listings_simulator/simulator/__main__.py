"""Publish prepared batches using environment settings or explicit CLI overrides."""
import argparse
from .helpers.settings import settings, publication_settings
from .helpers.publication import publish_day, publish_periodically


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--day', type=int, help='Publish one prepared day, for example 3.')
    mode.add_argument('--interval', type=float, help='Publish pending days in order, this many seconds apart, then exit.')
    args = parser.parse_args()
    config = settings()
    try:
        day, interval = publication_settings(args.day, args.interval)
        if day is not None:
            print(publish_day(day, config.data / 'arrival_examples', config.incoming))
        else:
            publish_periodically(interval, config.data / 'arrival_examples', config.incoming)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Publication interrupted. Published days remain available.\n')
    print('Published batches are available through the simulator API; consumer processing is separate.')


if __name__ == '__main__':
    main()
