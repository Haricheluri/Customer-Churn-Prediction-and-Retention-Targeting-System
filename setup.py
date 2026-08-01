from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(path:str)->List[str]:

    with open(path) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='Customer Churn Prediction and Retention Targeting System',
    version='0.01',
    author='Hari Haran',
    author_email='chelurihariharan@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)