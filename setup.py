import setuptools
import subprocess
from setuptools.extension import Extension
from setuptools.command.install import install
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


class InstallLocalPackage(install):
    def run(self):
        install.run(self)
        subprocess.call(
            "cd airutils; python3 setup.py build_ext --inplace", shell=True
        )


extensions = [
    Extension(
        'airutils.compute_overlap',
        ['airutils/compute_overlap.pyx']
    ),
]


setuptools.setup(
    name             = 'air-detector',
    version          = '0.0.2',
    description      = 'Keras implementation of Aerial Inspection RetinaNet object detector.',
    cmdclass         = {
        'build_ext': BuildExtension,
        'install':   InstallLocalPackage,
    },
    packages         = setuptools.find_packages(),
    install_requires = ['keras-resnet==0.2.0', 'six', 'scipy', 'h5py', 'numpy', 
                        'cython', 'Pillow', 'opencv-python', 'progressbar2',
                        'scikit-learn', 'matplotlib', 'filterpy'],
    extras_require   = {
        "gpu":        ["tensorflow==2.11.0"],
    },
    entry_points     = {
        'console_scripts': [
            'air-detect=detect:main',
            'air-train=keras_retinanet.keras_retinanet.bin.train:main',
            'air-evaluate=keras_retinanet.keras_retinanet.bin.evaluate:main',
            'air-infer=keras_retinanet.keras_retinanet.bin.infer:main',
            'air-debug=keras_retinanet.keras_retinanet.bin.debug:main',
            'air-convert-model=keras_retinanet.keras_retinanet.bin.convert_model:main',
        ],
    },
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',

        # Pick your license as you wish
        'License :: OSI Approved :: APACHE SOFTWARE LICENSE',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
    python_requires = '>=3.7, <3.8',
)
