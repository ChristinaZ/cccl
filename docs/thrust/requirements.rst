Requirements
============

All requirements are applicable to the ``main`` branch on GitHub. For
details on specific releases, please see the :ref:`changelog <thrust-module-releases-changelog>`.

Usage Requirements
------------------

To use the NVIDIA C++ Standard Library, you must meet the following
requirements.

System Software
~~~~~~~~~~~~~~~

Thrust and CUB require either the `NVIDIA HPC
SDK <https://developer.nvidia.com/hpc-sdk>`__ or the `CUDA
Toolkit <https://developer.nvidia.com/cuda-toolkit>`__.

Releases of Thrust and CUB are only tested against the latest releases
of NVHPC and CUDA. It may be possible to use newer version of Thrust and
CUB with an older NVHPC or CUDA installation by using a Thrust and CUB
release from GitHub, but please be aware this is not officially
supported.

C++ Dialects
~~~~~~~~~~~~

Thrust and CUB support the following C++ dialects:

-  C++11 (deprecated)
-  C++14
-  C++17

Compilers
~~~~~~~~~

Thrust and CUB support the following compilers when used in conjunction
with NVCC:

-  NVCC (latest version)
-  NVC++ (latest version)
-  GCC 5+
-  Clang 7+
-  MSVC 2019+ (19.20/16.0/14.20)

Unsupported versions may emit deprecation warnings, which can be
silenced by defining ``CCCL_IGNORE_DEPRECATED_COMPILER`` during
compilation.

Device Architectures
~~~~~~~~~~~~~~~~~~~~

Thrust and CUB support all NVIDIA device architectures since SM 35.

Host Architectures
~~~~~~~~~~~~~~~~~~

Thrust and CUB support the following host architectures:

-  aarch64.
-  x86-64.

Host Operating Systems
~~~~~~~~~~~~~~~~~~~~~~

Thrust and CUB support the following host operating systems:

-  Linux.
-  Windows.

Build and Test Requirements
---------------------------

To build and test Thrust and CUB yourself, you will need the following
in addition to the above requirements:

-  `CMake <https://cmake.org>`__.
