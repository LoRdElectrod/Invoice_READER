#!/usr/bin/env bash

# Update the apt package list
apt-get update

# Install system dependencies (like libgomp1)
apt-get install -y libgomp1

# Install all Python packages
pip install -r requirements.txt
