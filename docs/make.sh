#!/bin/bash

cd "$(dirname "$0")"
python -m sphinx -b html -D language=en . html
