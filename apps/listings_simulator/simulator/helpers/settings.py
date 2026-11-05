"""Local folders for the simulated upstream platform; no model or credentials required."""
from dataclasses import dataclass
import os
import math
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    data: Path

    @property
    def incoming(self):
        return self.data / "incoming"


def settings():
    default = Path(__file__).resolve().parents[4] / "local/listings_simulator"
    return Settings(Path(os.environ.get("LISTINGS_SIMULATOR_DATA", str(default))))


def publication_settings(day=None, interval=None):
    """Explicit arguments take precedence; validate only the selected mode."""
    if day is None and interval is None:
        mode = os.environ.get('SIMULATOR_MODE', 'once')
        if mode == 'once':
            try:
                day = int(os.environ.get('SIMULATOR_DAY', '1'))
            except ValueError:
                raise ValueError('SIMULATOR_DAY must be a positive integer.') from None
        elif mode == 'periodic':
            try:
                interval = float(os.environ.get('SIMULATOR_INTERVAL_SECONDS', '60'))
            except ValueError:
                raise ValueError('SIMULATOR_INTERVAL_SECONDS must be a positive finite number.') from None
        else:
            raise ValueError('SIMULATOR_MODE must be once or periodic.')
    if day is not None:
        if type(day) is not int or day <= 0:
            raise ValueError('Choose a positive integer day (SIMULATOR_DAY or --day).')
    elif isinstance(interval, bool) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Choose a positive finite interval (SIMULATOR_INTERVAL_SECONDS or --interval).')
    return day, interval
