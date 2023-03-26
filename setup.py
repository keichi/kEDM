from skbuild import setup

import versioneer

with open("README.md", "r") as f:
    readme = f.read()


setup(
    name="kedm",

    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    author="Keichi Takahashi",
    author_email="hello@keichi.dev",

    packages=["kedm"],
    package_dir={"kedm": "python/kedm"},

    cmake_args=["-DKEDM_ENABLE_PYTHON=ON",
                "-DKEDM_ENABLE_TESTS=OFF",
                "-DKEDM_ENABLE_EXECUTABLES=OFF",
                "-DKEDM_ENABLE_CPU=ON"
                ],

    url="https://github.com/keichi/kEDM",
    project_urls={
        "Documentation": "https://kedm.readthedocs.io/",
        "Source Code": "https://github.com/keichi/kEDM",
        "Bug Tracker": "https://github.com/keichi/kEDM/issues",
    },

    description="A high-performance implementation of the Empirical Dynamic"
                " Modeling (EDM) framework",

    long_description=readme,
    long_description_content_type="text/markdown",

    classifiers=[
        "License :: OSI Approved :: MIT License",
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
