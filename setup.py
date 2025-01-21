from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    """
    Function will return the list of requirements
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name             = 'mlops day2',
    version          = '0.0.1',
    author           = 'Bayuzen Ahmad',
    author_email     = 'bayuze19n@gmail.com',
    packages         = find_packages(include=['src', 'src.*']),
    # package_dir={'': 'src'},
    install_requires =get_requirements('requirements.txt')
    
)