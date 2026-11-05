"""Atomic publication of prepared CSV batches."""
import csv
import hashlib
import math
from pathlib import Path
import re
import shutil
import tempfile
import time


def publish_day(day: int, examples: Path, incoming: Path):
    if type(day) is not int or day < 1:
        raise ValueError('Choose a positive day number, for example --day 3.')
    source = examples / f'day_{day:03d}.csv'
    if not source.is_file():
        days = sorted(int(match.group(1)) for path in examples.glob('day_*.csv')
                      if (match := re.fullmatch(r'day_([0-9]+)\.csv', path.name)))
        available = ', '.join(map(str, days)) or 'none; place day_NNN.csv files in the arrival_examples folder'
        raise ValueError(f'Day {day} is not available. Available days: {available}.')
    destination = incoming / source.name
    if destination.exists():
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise ValueError(f'Day {day} is already published with different contents. Keep published days unchanged.')
        return f'Day {day} is already available.'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        next(reader, None)
        count = sum(1 for row in reader if row)
    incoming.mkdir(parents=True, exist_ok=True)
    # The simulator only lists .csv files. Rename after the full copy is closed.
    with tempfile.NamedTemporaryFile(dir=incoming, prefix=f'.day_{day:03d}-', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        # Recheck before publishing in case another command finished during the copy.
        if destination.exists():
            if temporary.read_bytes() != destination.read_bytes():
                raise ValueError(f'Day {day} is already published with different contents. Keep published days unchanged.')
            return f'Day {day} is already available.'
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return f'Day {day} published: {count} listings.'


def publish_periodically(interval, examples, incoming, *, wait=time.sleep, emit=print):
    """Publish the current prepared days in order; existing files record progress."""
    if isinstance(interval, bool) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Choose a positive finite interval in seconds.')
    days = sorted({int(match.group(1)) for path in examples.glob('day_*.csv')
                   if (match := re.fullmatch(r'day_([0-9]+)\.csv', path.name))})
    if not days:
        raise ValueError('No prepared days found. Place day_NNN.csv files in arrival_examples.')
    pending = []
    for day in days:
        if (incoming / f'day_{day:03d}.csv').exists():
            emit(publish_day(day, examples, incoming))
        else:
            pending.append(day)
    for index, day in enumerate(pending):
        if index:
            wait(interval)
        emit(publish_day(day, examples, incoming))
    emit(f'Publication complete: {len(pending)} new days. No prepared days remain.')
    return len(pending)
