import re
import itertools

def normalize_file_whitespace(lines):
    """remove initial and final whitespace on each line, replace any interal
    whitespace with one space, and remove trailing blank lines"""

    lines_out = []
    for l in lines.strip().splitlines():
        lines_out.append(re.sub('\s+', ' ', l.strip()))
    return '\n'.join(lines_out)

def filecmp_ignore_whitespace(f1, f2):
    """Compare two files ignoring all leading and trailing whitespace, amount of 
    whitespace within lines, and any trailing whitespace-only lines."""

    return (normalize_file_whitespace(open(f1).read()) ==
            normalize_file_whitespace(open(f2).read()))
