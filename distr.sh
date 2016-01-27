#!/bin/bash

if [ "$1" = "patch" ]; then
    echo "pull, patch, push, push --tags"
    git pull
    bumpversion patch
    git push
    git push --tags
fi
# upload to pypi
python setup.py register sdist upload

# build conda and upload

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

conda build -c mjirik -c SimpleITK .
conda convert -p all `conda build --output .`

binstar upload */*.tar.bz2

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

