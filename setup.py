# -*- coding: utf-8 -*-
import sys
import io

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install or upgrade setuptools or pip")
    sys.exit(1)

readme = io.open('README.rst', mode='r', encoding='utf-8').read()

setup(
    name='pmefm',
    version='0.1.dev',
    description='A package for science using numpy, matplotlib, readthedocs, etc.',
    long_description=readme,
    license='MIT',
    author='Ryan Dwyer',
    author_email='ryanpdwyer@gmail.com',
    url='https://github.com/ryanpdwyer/myscipkg4',
    zip_safe=False,
    # include_package_data=True,
    py_modules=['pmefm', 'lockin', 'phasekick', 'phasekickstan',
                'freqphasenoise'],
    # Add requirements here. If the requirement is difficult to install,
    # add to docs/conf.py MAGIC_MOCK, and .travis.yml 'conda install ...'
    install_requires=['numpy', 'scipy', 'matplotlib', "Click"],
    tests_require=['nose'],
    test_suite='nose.collector',
    entry_points='''
        [console_scripts]
        lockcli=lockin:lockcli
        firlockstate=lockin:firlockstate
        adiabatic_phasekick=lockin:adiabatic_phasekick_cli
        workup_adiabatic_avg=lockin:workup_adiabatic_avg
        adiabatic_report=phasekick:report_adiabatic_control_phase_corr_cli
        avg_df=phasekick:df_vs_t_cli
    ''',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
