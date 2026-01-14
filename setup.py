"""
/setup.py

Author: Jared Moore
Date: October, 2025
"""

import setuptools

with open("requirements.txt", "r", encoding="utf-8") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="transcripts",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="Transcript Helpers",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={
        "data": ["*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "process_chats = chatlog_processing_pipeline.commands:main",
        ]
    },
)
