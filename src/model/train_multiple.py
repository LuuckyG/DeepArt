#!/usr/bin/env python3
import sys
import subprocess

from io import StringIO

# Prepare multiple training experiments in batch file
with subprocess.Popen('run_training.bat', stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p, StringIO() as buf:
    for line in p.stdout:
        print(line, end='')
        buf.write(line)
    output = buf.getvalue()
rc = p.returncode
print(rc)
