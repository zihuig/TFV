from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['numpy>=1.19.5', 'torch>=1.1.0', 'scikit_learn>=0.21.3', 'pytorch-pretrained-bert==0.6.2', 
                    'transformers==2.6', 'multiprocess==0.70.11.1', 'tqdm==4.49.0',]


setup_requires = []

extras_require = {
}

classifiers = ["License :: OSI Approved :: GPL License"]

long_description = 'Table Fact Verification with Sampling and Structure-Aware Pretraining'

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='tfvsp',
    version=
    '0.0.1',  # please remember to edit __init__.py in response, once updating the version
    description='Table Fact Verification with Sampling and Structure-Aware Pretraining',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='cmk',
    author_email='3014428117@qq.com',
    packages=[
        package for package in find_packages()
        if package.startswith('tfvsp')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
