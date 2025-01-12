from setuptools import find_packages, setup
from typing import List

# Define the editable install identifier
HYPHEN_E = '-e .'

def get_requirements(path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of dependencies.
    Removes '-e .' if present.
    """
    requirements = []
    try:
        with open(path, 'r') as f:
            requirements = f.readlines()
            # Strip newline characters and whitespace
            requirements = [req.strip() for req in requirements]
            # Remove '-e .' if present
            if HYPHEN_E in requirements:
                requirements.remove(HYPHEN_E)
    except FileNotFoundError:
        print(f"Requirements file not found: {path}")
    return requirements

# Setup function for the package
setup(
    name='DSP01',
    version='0.0.1',
    author='Ayush Patidar',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
