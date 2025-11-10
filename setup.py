import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = os.path.realpath(os.path.dirname(__file__))


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        import torch

        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(
            os.path.realpath(self.build_lib), "metatomic_lj_test"
        )

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                build_dir,
                "--config",
                "Release",
                "--target",
                "install",
            ],
            check=True,
        )


if __name__ == "__main__":
    setup(
        ext_modules=[
            Extension(name="metatomic_lj_test", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
        },
        package_data={
            "metatomic_lj_test": [
                "metatomic_lj_test/lib/*",
            ]
        },
    )
