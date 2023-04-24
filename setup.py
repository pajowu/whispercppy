import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from sysconfig import get_path

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _build_ext
from setuptools import Extension, setup

WHISPER_ENABLE_COREML = False

__version__ = "0.0.2"


class CopyWhisperDummyExtension(Extension):
    pass


class build_ext(_build_ext):
    def run(self):
        # Run CMake to build the libwhisper library
        cmake_dir = os.path.realpath(os.path.join("external", "whisper.cpp"))
        pkg_build_dir = os.path.realpath("build")
        os.makedirs(pkg_build_dir, exist_ok=True)

        extra_cmake_flags = []
        if WHISPER_ENABLE_COREML:
            extra_cmake_flags.append("-DWHISPER_COREML=1")

        # Based on the scikit build tooling
        archflags = os.environ.get("ARCHFLAGS")
        if archflags is not None:
            archs = ";".join(set(archflags.split()) & {"x86_64", "arm64"})
            extra_cmake_flags.append("-DCMAKE_OSX_ARCHITECTURES:STRING=%s" % archs)

        subprocess.check_call(
            [
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                *extra_cmake_flags,
                "-Wno-dev",
                cmake_dir,
            ],
            cwd=pkg_build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "whisper"], cwd=pkg_build_dir
        )

        filtered_ext = []
        for ext in self.extensions:
            if isinstance(ext, CopyWhisperDummyExtension):
                ext_dir = (
                    Path(self.get_ext_fullpath(ext.name)).parent.absolute() / ext.name
                )

                for lib_file in find_library_paths():
                    print("copying", lib_file, "to", ext_dir)
                    shutil.copyfile(
                        lib_file,
                        ext_dir / os.path.basename(lib_file),
                    )
            else:
                filtered_ext.append(ext)
        self.extensions = filtered_ext
        super().run()


def find_library_paths():
    yield from glob("build/libwhisper.*")


lib_path = get_path("platlib")
ext_modules = [
    CopyWhisperDummyExtension("whispercppy", []),
    Pybind11Extension(
        "whispercppy.api_cpp2py_export",
        [
            os.path.join("src", "api_cpp2py_export.cc"),
            os.path.join("src", "params.cc"),
            os.path.join("src", "context.cc"),
            os.path.join("external", "whisper.cpp", "examples", "common.cpp"),
        ],
        include_dirs=[
            "src",
            os.path.join("external", "whisper.cpp"),
            os.path.join(
                "external",
                "whisper.cpp",
                "examples",
            ),
        ],
        library_dirs=["./build"],
        libraries=["whisper"],
        runtime_library_dirs=["@loader_path", "$ORIGIN"],
    ),
]


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="whispercppy",
    version=__version__,
    author="pajowu",
    author_email="git@ca.pajowu.de",
    url="https://github.com/pajowu/whispercppy",
    description="Python bindings for whisper.cpp",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=["whispercppy"],
    package_dir={"whispercppy": "src"},
    python_requires=">=3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={"": ["src/*.h"]},
)
