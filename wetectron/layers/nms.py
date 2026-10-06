# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from wetectron import _C
try:
    from apex import amp
except ImportError:
    class _AmpFallback(object):
        @staticmethod
        def float_function(func):
            return func
    amp = _AmpFallback()

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
