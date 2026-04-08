#!/usr/bin/env python3

from roop.core import app as core_app
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--execution-provider', default='cuda', help='Execution provider: cpu or cuda')
args = parser.parse_args()
import roop.config.globals
roop.config.globals.execution_providers = [args.execution_provider + 'ExecutionProvider']

if __name__ == '__main__':
    core_app.run()
