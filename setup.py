#!/usr/bin/env python

from pathlib import Path
import setuptools

requirements_file = Path(__file__).parent / 'requirements.txt'
with requirements_file.open('r') as fh:
    requirement_lines = fh.readlines()

dev_requirements_file = Path(__file__).parent / 'requirements-dev.txt'
with dev_requirements_file.open('r') as fh:
    dev_requirement_lines = fh.readlines()

setuptools.setup(
    name='some_package',
    description='Package description',
    url='https://github.com/org_name/repo_name',
    packages=setuptools.find_packages(),
    python_requires='>3.7',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'some-cli = some_package.cli.some_cli:main',
        ]
    },
    install_requires=requirement_lines,
    extras_require={
       'dev': dev_requirement_lines
    }
)
