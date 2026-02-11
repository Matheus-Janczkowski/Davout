from setuptools import setup, find_packages

with open("README.md", 'r') as f:

    long_description = f.read()

setup(
    name='Davout',
    version='0.1',
    package_dir={"": "source"},
    packages=find_packages(where="source"),
    long_description=long_description,
    author="Matheus Janczkowski",
    author_email="matheusj2009@hotmail.com",
    install_requires=['numpy', 'scipy', 'matplotlib'],
    include_package_data=True,
    package_data={'Davout': ['LaTeXUtilities/*.sty'], },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        # Add more classifiers as needed
    ],
    python_requires='>=3.10',  # Specify the required Python version
)
