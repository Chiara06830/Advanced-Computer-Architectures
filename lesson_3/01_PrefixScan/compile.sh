#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_62 "$DIR"/PrefixScan.cu -I"$DIR"/include -o prefixscan
