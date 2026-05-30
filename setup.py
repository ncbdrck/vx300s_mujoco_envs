#!/usr/bin/env python

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'vx300s_mujoco_reach',
        'vx300s_mujoco_reach.robot_envs',
        'vx300s_mujoco_reach.task_envs',
        'vx300s_mujoco_reach.task_envs.reach',
    ],
    package_dir={'': 'src'},
)

setup(**setup_args)
