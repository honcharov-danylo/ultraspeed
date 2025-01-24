# setup.py
from setuptools import setup, Extension
import numpy

module = Extension(
    'ultraspeed',
    sources=['ultraspeed_mod.c', 'ultraspeed.c'],
    include_dirs=[numpy.get_include(), '.', '/usr/local/include','/usr/local/include/blis/'],  # Include current directory for header files
    extra_compile_args=['-O2', '-march=native', '-fopenmp', '-mno-avx512f','-ffast-math'],
    extra_link_args=['-fopenmp','-lblis-mt', '-Wl,-rpath,/usr/local/lib/'],
    library_dirs=[
        '/usr/local/lib/',      # Adjust this path
    ],
    libraries=[
        'blis-mt'
    ],
)


# -march=native -mno-avx512f -Lbuild/src -I/usr/local/include/blis/ -L/usr/local/lib/  -shared -o errors.so fast_error.c -lmatmul -lblis-mt -mfma


setup(
    name='ultraspeed',
    version='0.1',
    description='Python interface for optimized matrix multiplication',
    ext_modules=[module],
)