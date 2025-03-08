from setuptools import setup

setup(
    name="deeplogit",
    version="0.1.2",
    description="Mixed Logit Estimation with Text and Image Embeddings Extracted Using Deep Learning Models",
    url="https://github.com/deep-logit-demand/deeplogit",
    author="",
    author_email="",
    license="GPL-3.0",
    packages=["deeplogit"],
    install_requires=[
        "numpy",
        "xlogit",
        "pandas",
        "tensorflow",
        "tensorflow_hub",
        "torch",
        "sentence-transformers",
        "keras",
        "scikit-learn",
        "nltk",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
