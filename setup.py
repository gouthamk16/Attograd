from setuptools import setup, find_packages

setup(
    name='attograd',
    version='0.1.0',
    description='A lightweight neural network framework built from scratch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Goutham',
    author_email='',
    url='https://github.com/gouthamk16/attograd',
    packages=find_packages(include=['attograd', 'attograd.*']),
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.4.0',
        'graphviz>=0.16',
    ],
    extras_require={
        'cuda': ['cupy>=10.0.0'],
        'dev': [
            'pytest',
            'pytest-cov',
            'flake8',
            'black',
            'sphinx',
            'sphinx_rtd_theme',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
