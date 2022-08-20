

from setuptools import setup, find_packages

setup(name="popdyn",
      packages=find_packages(),
      description="population dynamics in historical maps",
      install_requires=['numpy', 'scipy', 'pandas',],
      author="Tamas Norbert Varga",
      author_email="T.Varga@physik.lmu.de",
      version="0.1")