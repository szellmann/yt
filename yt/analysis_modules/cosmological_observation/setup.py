#!/usr/bin/env python
import setuptools
import os
import sys
import os.path


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cosmological_observation', parent_package, top_path)
    config.make_config_py()  # installs __config__.py
    #config.make_svn_version_py()
    config.add_subpackage("light_cone")
    config.add_subpackage("light_ray")
    return config