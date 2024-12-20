from setuptools import setup, find_packages
import os


# import ipdb; ipdb.set_trace()


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


# extra_files = package_files('nsfr/data') + package_files('nsfr/lark')
extra_files = package_files('neumann/lark')
# import ipdb; ipdb.set_trace()

setup(
    name='neumann',
    version='0.1.0',
    author='Hikaru Shindo',
    author_email='hikisan.gouv',
    packages=find_packages(),
    package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'nudge'},
    url='tba',
    description='Neuro Symbolic Message Passing Forward Reasonner',
    long_description=open('README.md').read(),
    install_requires=[
        "matplotlib",
        "numpy",
        "seaborn",
        "setuptools",
#        "torch",
        "tqdm",
	"lark",
    ],
)