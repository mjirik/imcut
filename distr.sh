#!/bin/bash

# upload to pypi
python setup.py register sdist upload

# build conda and upload

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

conda build .
conda convert -p all `conda build --output .`

binstar upload */*.tar.bz2

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

