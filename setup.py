from setuptools import setup, find_packages

setup(
    name="EEDM",
    version="0.1.0",
    description="Code package to compute the EigenEnergy Decomposition Method.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Axel Raboonik",
    author_email="raboonik@gmail.com",
    url="https://github.com/raboonik/EEDM",
    project_urls={
        "Documentation": "https://github.com/raboonik/EEDM#readme",
        "Journal Article": "https://iopscience.iop.org/article/10.3847/1538-4357/adc917",
        "Source Code": "https://github.com/raboonik/EEDM",
    },
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "mpi4py",
        "scipy>=1.4.0",
        "h5py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or change if you use a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "eedm = eedm.__main__:main"
        ]
    },
)