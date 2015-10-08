"""Run all of the available tests.
This will run all files that have a name with the format *_test.py.
It will print out each file name and whether all of the tests passed or not.
"""
import os
import glob

# --- Run each of the files independently.
for f in glob.glob('*_test.py'):
    fin,fout,ferr = os.popen3('python %s'%f)
    serr = ferr.readlines()
    if serr[-1] == 'OK\n':
        print('%s OK\n'%f)
    else:
        print('\033[1;31m%s\033[0m'%(f))
        for e in serr[:-1]:
            print(e)
        # --- Use ASCII codes to print this is red
        print('\033[1;31m%s %s\033[0m'%(f,serr[-1]))

