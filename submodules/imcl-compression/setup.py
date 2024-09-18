from pathlib import Path
from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


cwd = Path(__file__).resolve().parent

package_name = "data_compression"
version = "0.3"

install_requires = [
    "torch",
    "torchvision"
]

def get_extensions():
    ext_dirs = cwd / package_name / "cpp_exts"
    ext_modules = []

    # Add rANS module
    rans_lib_dir = ext_dirs / "ryg_rans"
    rans_ext_dir = ext_dirs / "rans"

    extra_compile_args = ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.rans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            extra_compile_args=extra_compile_args,
        )
    )
    return ext_modules


setup(
    name=package_name,
    version=version,
    description="Data compression in Pytorch.",
    zip_safe=False,
    install_requires=install_requires,
    license="Apache-2",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
)
