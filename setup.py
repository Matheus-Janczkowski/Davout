from setuptools import setup, find_packages

setup(
    name='Grante',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'
    ],
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
