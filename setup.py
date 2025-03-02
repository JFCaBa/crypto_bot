#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot
Installation script
"""

from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Remove comments and empty lines
requirements = [r for r in requirements if r and not r.startswith('#')]

setup(
    name="cryptobot",
    version="1.0.0",
    description="Cryptocurrency Trading Bot",
    author="CryptoBot Team",
    author_email="info@cryptobot.example",
    url="https://github.com/cryptobot-team/cryptobot",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cryptobot=cryptobot.ui.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
)