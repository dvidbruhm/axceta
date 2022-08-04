from pathlib import Path
from rich import print


def assert_csv(file: Path):
    assert str(file.absolute())[-4:] == ".csv", "Input file should be a csv"
