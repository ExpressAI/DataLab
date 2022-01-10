#!/usr/bin/env python
from argparse import ArgumentParser

from datalab.commands.convert import ConvertCommand
from datalab.commands.dummy_data import DummyDataCommand
from datalab.commands.env import EnvironmentCommand
from datalab.commands.run_beam import RunBeamCommand
from datalab.commands.test import TestCommand
from datalab.utils.logging import set_verbosity_info


def main():
    parser = ArgumentParser("HuggingFace Datasets CLI tool", usage="datalab-cli <command> [<args>]")
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
