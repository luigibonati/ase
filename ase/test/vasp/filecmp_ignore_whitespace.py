import re
import itertools

def filecmp_ignore_whitespace(f1, f2, fillvallue=None):
    with open(f1) as ff1, open(f2) as ff2:
        # check for mismatching lines
        for l1, l2 in itertools.zip_longest(ff1, ff2):
            if l1 is None or l2 is None:
                break
            if re.sub(r'\s+', ' ', l1.strip()) != re.sub(r'\s+', ' ', l2.strip()):
                return False
        # check for additional non-blank lines after f1 finished
        if l1 is None:
            if len(l2.strip()) > 0:
                return False
            for l2 in ff2:
                if len(l2.strip()) > 0:
                    return False
        # check for additional non-blank lines after f2 finished
        if l2 is None:
            if len(l1.strip()) > 0:
                return False
            for l1 in ff1:
                if len(l1.strip()) > 0:
                    return False

    return True
