#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from quilt3 import Package

class QuiltUploader():
    """
    Upload model outputs to Quilt
    """
    @staticmethod
    def upload_to_quilt(model_name, model_iteration, comment):
        """
        Upload model outputs
        """
        print("quilt upload = {} {} : {}".format(
            model_name, model_iteration, comment))
        p = Package()
        current_dir = os.getcwd()
        p.set("parameters.script",
            "{}_batch.script".format(model_name))
        p.set("{}.py".format(model_name),
            "{}.py".format(model_name))
        p.set("{}_utility.py".format(model_name),
            "{}_utility.py".format(model_name))
        p.set_dir("logs", "logs")
        p.set_dir("checkpoints", "checkpoints")
        p.set_dir("trajectory", "trajectory")
        p.set_dir("visualization", "visualization")
        p.push("aics/simularium_readdy_{}".format(model_name),
            "s3://allencell-internal-quilt", message="{} : {}".format(
                model_iteration, comment))
