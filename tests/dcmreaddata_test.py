#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

# import funkcí z jiného adresáře
import sys
import os.path
import unittest
import scipy
import numpy as np
# import imcut.dcmreaddata

import logging
logger = logging.getLogger(__name__)

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))

from nose.plugins.attrib import attr
from nose.tools import raises
from imcut import graph

class DcmReadDataTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     if sys.version_info.major < 3:
    #         cls.assertCountEqual = cls.assertItemsEqual
    def test_dcmread(self):
        pass
