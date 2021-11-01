# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "pylossmap_ml",
    "pylossmap_ml.fetching",
    "pylossmap_ml.preprocessing",
    "pylossmap_ml.training",
]

package_data = {"": ["*"]}

install_requires = [
    "matplotlib>=3.4.3,<4.0.0",
    "numpy>=1.19.0,<2.0.0",
    "pandas>=1.2.3,<2.0.0",
    "pylossmap @ git+https://github.com/loiccoyle/pylossmap@master",
    "pytimber>=3.2.1,<4.0.0",
    "tables>=3.6.1,<4.0.0",
    "tensorflow>=2.6.0,<3.0.0",
    "tqdm>=4.59.0,<5.0.0",
]

entry_points = {"console_scripts": ["lm-fetch = pylossmap_ml.fetching.__main__:main"]}

setup_kwargs = {
    "name": "pylossmap-ml",
    "version": "0.1.0",
    "description": "",
    "long_description": None,
    "author": "Loic Coyle",
    "author_email": "loic.coyle@hotmail.fr",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "python_requires": ">=3.7.1,<4.0.0",
}


setup(**setup_kwargs)
