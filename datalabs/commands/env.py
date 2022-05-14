# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the DataLab Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import platform

import pyarrow

from datalabs import __version__ as version
from datalabs.commands import BaseDatasetsCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser(
            "env", help="Print relevant system environment info."
        )
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        info = {
            "`datalab` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyArrow version": pyarrow.__version__,
        }

        print("\nCopy-and-paste the text below in your GitHub issue.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
