#!/bin/sh

set -ex

wget -q https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz
cargo run
