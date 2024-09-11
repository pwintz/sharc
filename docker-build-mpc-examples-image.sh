#!/usr/bin/env bash
# A script to abbreviate building 

docker image rm mpc-examples
docker build --target mpc-examples --tag mpc-examples $@ .
