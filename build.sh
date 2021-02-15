#!/bin/bash

rm -rf build && rm -rf dist
pip install -r dev_requirements.txt || exit 1
pip install -r requirements.txt || exit 1
pytest || exit 1
python setup.py sdist bdist_wheel || exit 1
