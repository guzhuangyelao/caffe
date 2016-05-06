#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/12306/12306_solver.prototxt
