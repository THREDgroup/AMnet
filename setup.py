from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='WAnet',
    version='0.0.1',
    long_description=readme,
    author='Christopher McComb',
    author_email='chris.c.mccomb@gmail.com',
    license=license,
)
