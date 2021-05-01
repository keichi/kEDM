#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
brew install hdf5
