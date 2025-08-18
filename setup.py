from setuptools import setup, find_packages

with open("README.md", 'r') as f:

    long_description = f.read()

setup(
    name='Grante',
    version='0.1',
    package_dir={"": "source"},
    packages=find_packages(where="source"),
    long_description=long_description,
    author="Matheus Janczkowski",
    author_email="matheusj2009@hotmail.com",
    install_requires=['numpy', 'scipy', 'matplotlib'],
    include_package_data=True,
    package_data={
        # Include additional files like images, text files, etc.
        # '': ['*.txt', '*.rst'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        # Add more classifiers as needed
    ],
    python_requires='>=3.6',  # Specify the required Python version
)
