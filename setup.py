from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crash-detection-system",
    version="1.0.0",
    author="Your Name",
    author_email="contact@crashdetection.ai",
    description="Intelligent Multimodal Edge-AI Crash Detection & Prevention System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/crash-detection-system",
    packages=find_packages(exclude=["tests", "docs", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.1",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "jetson": [
            "jetson-stats>=4.2.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "crash-detect=inference.pipeline:main",
            "crash-train=training.train_crash_predictor:main",
            "crash-api=api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)
