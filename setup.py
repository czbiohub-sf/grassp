#!/usr/bin/env python

import setuptools

setuptools.setup(
    name='package_name',
    description='Package description',
    url='https://github.com/org_name/repo_name',
    packages=setuptools.find_packages(),
    python_requires='>3.7',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'cmd = package_name.cli.some_cli_module:main',
        ]
    }
)
