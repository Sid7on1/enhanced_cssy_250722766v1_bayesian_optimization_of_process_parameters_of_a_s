import os
import sys
import logging
import setuptools
from setuptools import setup, find_packages
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define constants
PROJECT_NAME = "enhanced_cs.SY_2507.22766v1_Bayesian_Optimization_of_Process_Parameters_of_a_S"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Bayesian Optimization of Process Parameters of a Sensor-Based Sorting System"

# Define dependencies
DEPENDENCIES = {
    "required": [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn"
    ],
    "optional": [
        "opencv-python",
        "scikit-image"
    ]
}

# Define setup function
def setup_package() -> None:
    try:
        # Create setup configuration
        setup(
            name=PROJECT_NAME,
            version=PROJECT_VERSION,
            description=PROJECT_DESCRIPTION,
            long_description=open("README.md").read(),
            long_description_content_type="text/markdown",
            author="Felix Kronenwett, Georg Maier, Thomas Länge",
            author_email="felix.kronenwett@iosb.fraunhofer.de, georg.maier@iosb.fraunhofer.de, thomas.laengle@iosb.fraunhofer.de",
            url="https://github.com/felixkronenwett/enhanced_cs.SY_2507.22766v1_Bayesian_Optimization_of_Process_Parameters_of_a_S",
            packages=find_packages(),
            install_requires=DEPENDENCIES["required"],
            extras_require=DEPENDENCIES["optional"],
            classifiers=[
                "Development Status :: 5 - Production/Stable",
                "Intended Audience :: Developers",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10"
            ],
            keywords=["Bayesian Optimization", "Process Parameters", "Sensor-Based Sorting System"],
            project_urls={
                "Documentation": "https://github.com/felixkronenwett/enhanced_cs.SY_2507.22766v1_Bayesian_Optimization_of_Process_Parameters_of_a_S",
                "Source Code": "https://github.com/felixkronenwett/enhanced_cs.SY_2507.22766v1_Bayesian_Optimization_of_Process_Parameters_of_a_S",
                "Bug Tracker": "https://github.com/felixkronenwett/enhanced_cs.SY_2507.22766v1_Bayesian_Optimization_of_Process_Parameters_of_a_S/issues"
            }
        )
    except Exception as e:
        logging.error(f"Error setting up package: {e}")
        sys.exit(1)

# Run setup function
if __name__ == "__main__":
    setup_package()