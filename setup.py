from setuptools import setup, find_packages


def read(name):
    with open(name, "r") as fd:
        return fd.read()


setup(
    name="StablePacking2D",
    version="0.0.0",

    author="Qianqing Wang",
    author_email="qianqing.wang@epfl.ch",

    packages=find_packages(),

)
