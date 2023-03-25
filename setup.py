
from setuptools import find_packages, setup
from typing import List

flag='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as reader:
        requirements=reader.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if flag in requirements:
            requirements.remove(flag) 

    return requirements


setup(
    name='TwiiterAndRedditSentimentModel',
    version='0.0.1',
    author='Deepak',
    author_email='deepak170602@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)