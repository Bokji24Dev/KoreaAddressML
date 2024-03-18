#!/usr/bin/env python
from setuptools import setup, find_packages

def install():
    required = []
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        for req in requirements:
            p = req.split('==')
            required.append(p[0])
    desc = '한국 주소 예측 머신러닝 모델'
    setup(
        name='py-koraddr',
        version='1.0.0',
        description=desc,
        long_description=desc,
        author='KeunSeok Im, HoGyeom Lee',
        author_email='mineru664500@gmail.com',
        url='https://github.com/Bokji24Dev/KoreaAddressML',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Korea Address',
            'Intended Audience :: Classification',
            'License :: Apache',
            'Operating System :: POSIX',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Korea Address',
            'Topic :: Classification',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11'
        ],
        packages=find_packages(),
        install_requires=required,
    )


if __name__ == "__main__":
    install()