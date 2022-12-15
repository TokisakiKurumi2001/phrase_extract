import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PhrExt',
    packages=['PhrExt'],
    version='0.1.0',
    description='Extract phrase from English sentence such as noun, verb, preposition based on NLP task word chunking and HuggingFace library',
    author='Le Minh Khoi',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["PhrExt"],
    install_requires=[]
)