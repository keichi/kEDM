# Copyright Spack Project Developers.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import glob
import os

from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage


class Kedm(CMakePackage, CudaPackage):
    homepage = "https://github.com/freifrauvonbleifrei/kEDM"
    git = "https://github.com/freifrauvonbleifrei/kEDM.git"

    version("master", branch="master")

    variant("cpu", default=True, description="Enable CPU (OpenMP) backend")
    variant("cuda", default=False, description="Enable GPU (CUDA) backend")
    variant("mpi", default=False, description="Enable MPI support")
    variant("python", default=True, description="Build Python bindings")
    variant("executables", default=True, description="Build executables")
    variant("tests", default=False, description="Build unit tests")
    variant("likwid", default=False, description="Enable LIKWID performance counters")
    variant("scratch_memory", default=True, description="Use Kokkos scratch memory")
    variant("simd_primitives", default=True, description="Use Kokkos SIMD primitives")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("cmake@3.22:", type="build")

    # Core deps (external mode expects these discoverable by CMake)
    depends_on("kokkos@5.0.0: +openmp", when="+cpu")
    depends_on("kokkos@5.0.0: +cuda", when="+cuda")

    depends_on("boost")  # provides Boost headers used by Boost.Math

    # not currently in spack, but could be added
    # depends_on("argh")
    # depends_on("pcg-cpp")

    # BLAS/LAPACK vs CUDA BLAS
    depends_on("lapack", when="~cuda")
    depends_on("cuda", when="+cuda")

    # MPI
    depends_on("mpi", when="+mpi")

    # Executables: HighFive/HDF5
    depends_on("highfive", when="+executables")
    depends_on("hdf5 +cxx", when="+executables")
    depends_on("hdf5 +cxx +mpi", when="+executables+mpi")

    # Optional LIKWID
    depends_on("likwid", when="+likwid")

    # Python bindings (use C++ pybind11 package, not only the Python module)
    extends("python", when="+python")
    depends_on("python@3.8:", when="+python")
    depends_on("pybind11", when="+python")

    # Tests
    depends_on("doctest", when="+tests")

    def cmake_args(self):
        args = []

        # Tell upstream to use find_package/find_path instead of FetchContent
        args.append(self.define("KEDM_USE_EXTERNAL_DEPS", True))

        args += [
            self.define("KEDM_ENABLE_CPU", "+cpu" in self.spec),
            self.define("KEDM_ENABLE_GPU", "+cuda" in self.spec),
            self.define("KEDM_ENABLE_MPI", "+mpi" in self.spec),
            self.define("KEDM_ENABLE_PYTHON", "+python" in self.spec),
            self.define("KEDM_ENABLE_EXECUTABLES", "+executables" in self.spec),
            self.define("KEDM_ENABLE_TESTS", "+tests" in self.spec),
            self.define("KEDM_ENABLE_LIKWID", "+likwid" in self.spec),
            self.define("KEDM_ENABLE_SCRATCH_MEMORY", "+scratch_memory" in self.spec),
            self.define("KEDM_ENABLE_SIMD_PRIMITIVES", "+simd_primitives" in self.spec),
        ]

        return args

    def install(self, spec, prefix):
        # Run upstream install (installs python module if enabled)
        super().install(spec, prefix)

        # Install library artifacts (upstream doesn't define install rules for kedm lib)
        mkdirp(prefix.lib)

        candidates = []
        candidates += glob.glob(os.path.join(self.build_directory, "libkedm.*"))
        candidates += glob.glob(os.path.join(self.build_directory, "kedm.*"))
        for f in sorted(set(candidates)):
            if os.path.isfile(f) and (
                f.endswith(".a")
                or f.endswith(".so")
                or ".so." in f
                or f.endswith(".dylib")
            ):
                install(f, prefix.lib)

        # Install executables if built
        if "+executables" in spec:
            mkdirp(prefix.bin)
            for exe in [
                "edm-xmap",
                "knn-bench",
                "lookup-bench",
                "smap-bench",
                "partial-sort-bench",
                "edm-xmap-mpi",
            ]:
                p = os.path.join(self.build_directory, exe)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    install(p, prefix.bin)

        # Install headers
        mkdirp(prefix.include)
        incdir = os.path.join(prefix.include, "kedm")
        mkdirp(incdir)
        for pattern in ["*.hpp", "*.h", "*.cuh", "*.hh"]:
            for hdr in glob.glob(os.path.join(self.stage.source_path, "src", pattern)):
                install(hdr, incdir)

        # License files (if present)
        for fname in ["LICENSE", "LICENSE-THIRD-PARTY"]:
            f = os.path.join(self.stage.source_path, fname)
            if os.path.isfile(f):
                install(f, prefix)

    def test(self):
        assert os.path.isdir(self.prefix.lib)
