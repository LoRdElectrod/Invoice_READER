#!/usr/bin/env bash

# Install system-level dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
