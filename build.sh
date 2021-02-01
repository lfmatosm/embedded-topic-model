#!/bin/bash

pip install -r dev_requirements.txt
pip install -r requirements.txt
pytest
python setup.py sdist bdist_wheel
