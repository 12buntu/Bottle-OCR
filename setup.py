"""
Setup script for Bottle OCR module.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="bottle-ocr",
    version="0.1.0",
    description="Flask module for recognizing text from round bottles using video capture",
    author="DDomLab",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["../static/*.html"],
    },
    install_requires=[
        "flask>=2.0.0",
        "flask-cors>=3.0.10",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pytesseract>=0.3.10",
    ],
    extras_require={
        "easyocr": ["easyocr>=1.7.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
