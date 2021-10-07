import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="src",
    version="0.0.1",
    author="vishalbansal-1650",
    description="It is an implmentation of ANN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishalbansal-1650/ANN-Implementation",
    author_email="vishalbansal1650@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)
