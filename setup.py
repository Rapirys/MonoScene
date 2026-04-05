from pathlib import Path
from setuptools import setup
from setuptools import find_packages

# for install, do: pip install -ve .

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="monoscene",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=requirements,
)
