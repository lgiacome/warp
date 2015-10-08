"""Run all of the available tests.
This will run all files that have a name with the format *_test.py.
It will print out each file name and whether all of the tests passed or not.
"""
import glob
from subprocess import Popen, PIPE

try:
    from termcolor import colored
except:
    def colored(a, b):
        return a

# --- Run each of the files independently.
for f in glob.glob('*_test.py'):
    p = Popen(["python", f], stdout=PIPE, stderr=PIPE, close_fds=True)
    serr = p.stderr.readlines()
    if serr[-1] == 'OK\n':
        print('%s %s'%(f, colored('OK', 'green')))
    else:
        print(colored(f, 'red'))
        for e in serr[:-1]:
            print(e)
        # --- Use ASCII codes to print this is red
        print('%s %s'%(colored(f, 'red'), colored(serr[-1], 'red')))

