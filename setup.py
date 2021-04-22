from skbuild import setup


with open("README.md", "r") as f:
    readme = f.read()


setup(
    name="kedm",

    version="0.1.0",

    author="Keichi Takahashi",
    author_email="keichi.t@me.com",

    packages=["kedm"],
    package_dir={"kedm": "python/kedm"},

    cmake_args=["-DUSE_PYTHON:BOOL=ON"],

    url="https://github.com/keichi/kEDM",
    project_urls={
        "Documentation": "https://kedm.readthedocs.io/",
        "Source Code": "https://github.com/keichi/kEDM",
        "Bug Tracker": "https://github.com/keichi/kEDM/issues",
    },

    description="kEDM is a high-performance implementation of the Empirical"
                "Dynamical Modeling (EDM) framework",

    long_description=readme,

    classifiers=[
        "OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],

    license="MIT",

    install_requires=["numpy>=1.7.0"],
    extras_require={
        "test": ["pytest>=6.2.0"],
    },
)
