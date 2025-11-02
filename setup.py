from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import glob

binding_sources = glob.glob("binding/*.cpp")

core_sources = glob.glob("source/*.cpp")

ext_modules = [
    Pybind11Extension(
        "HSmodule",  # module name
        sources=binding_sources + core_sources, # add .cpp file
        include_dirs=[
            pybind11.get_include(),
            "header"
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=["-O3", "-Ofast", "-march=native", "-DNDEBUG", "-std=c++17"],
    ),
]

setup(
    name="HSmodule",
    version="0.1",
    author="Gia Khanh",
    description="Hash + Search for duplication_text + bloom_filter_prefilter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
