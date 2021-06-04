#!/bin/bash
# copied from stable-baselines3
python3 -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v
