#!/usr/bin/env python
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

from datalabs.commands.convert import ConvertCommand
from datalabs.commands.dummy_data import DummyDataCommand
from datalabs.commands.env import EnvironmentCommand
from datalabs.commands.run_beam import RunBeamCommand
from datalabs.commands.test import TestCommand
from datalabs.utils.logging import set_verbosity_info


def main():
    parser = ArgumentParser("DataLabs CLI tool",
                            usage="datalab-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="datalab-cli command helpers")
    set_verbosity_info()

    # Register commands
    ConvertCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    TestCommand.register_subcommand(commands_parser)
    RunBeamCommand.register_subcommand(commands_parser)
    DummyDataCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
