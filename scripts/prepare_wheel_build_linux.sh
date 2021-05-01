#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
yum install -y openblas-devel hdf5-devel
