# setup.py
# Setup installation for the application
# install with:
# python3 -m pip install -e ".[dev]"

from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages: list[str] = []

dev_packages: list[str] = [
    "rope"
]

docs_packages: list[str] = []

setup(
    name="ultrasound",
    version="0.1",
    description="Noise reduction on ultrasonic sensor data to measure quantity of grains in a silo.",
    author="David Abraham",
    python_requires=">=3.9",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
    entry_points={
        'console_scripts': [
            'us=us.main:app',
        ],
    },
)
