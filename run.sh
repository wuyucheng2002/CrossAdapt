#!/bin/bash
python3 run.py --mode teacher;
python3 run.py --mode tskd --replay true --sample_selection biased_sample --sample_ratio 0.1;