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

        # ==================================================================== #
        # HACK: Torch cmake build system has a hard time finding CuDNN, so we
        # help it by pointing it to the right files

        # First try using the `nvidia.cudnn` package (dependency of torch on PyPI)
        try:
            import nvidia.cudnn

            cudnn_root = os.path.dirname(nvidia.cudnn.__file__)
        except ImportError:
            # Otherwise try to find CuDNN inside PyTorch itself
            cudnn_root = os.path.join(torch.utils.cmake_prefix_path, "..", "..")

            cudnn_version = os.path.join(cudnn_root, "include", "cudnn_version.h")
            if not os.path.exists(cudnn_version):
                # create a minimal cudnn_version.h (with a made-up version),
                # because it is not bundled together with the CuDNN shared
                # library in PyTorch conda distribution, see
                # https://github.com/pytorch/pytorch/issues/47743
                with open(cudnn_version, "w") as fd:
                    fd.write("#define CUDNN_MAJOR 8\n")
                    fd.write("#define CUDNN_MINOR 5\n")
                    fd.write("#define CUDNN_PATCHLEVEL 0\n")

        cmake_options.append(f"-DCUDNN_INCLUDE_DIR={cudnn_root}/include")
        cmake_options.append(f"-DCUDNN_LIBRARY={cudnn_root}/lib")
        # do not warn if the two variables above aren't used
        cmake_options.append("--no-warn-unused-cli")

        # end of HACK
        # ==================================================================== #

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
