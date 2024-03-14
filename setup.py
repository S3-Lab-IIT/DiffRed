from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='diffred',
    packages=['DiffRed'],
    author='Prarabdh Shukla',
    author_email='prarabdh.10@gmail.com',
    description='Official Implementation of "DiffRed: Dimensionality Reduction guided by stable rank"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/S3-Lab-IIT/DiffRed/tree/master',
    project_urls={
        'Homepage': 'https://github.com/S3-Lab-IIT/DiffRed/tree/master'
    },
    download_url='https://github.com/S3-Lab-IIT/DiffRed/archive/refs/tags/v0.1.0.tar.gz',
    install_requires=install_requires
)