from __future__ import print_function

import argparse
import os
import sys
import subprocess

def main():
    return_value = 0
    filename = sys.argv[1]
    dir_py = os.path.dirname(filename)
    # subprocess.run([f'mkdir {dir_py}'], shell=True)
    subprocess.run([f'git add {dir_py}'], shell=True)
    fpy = os.path.join(dir_py,  f"{os.path.splitext(os.path.basename(filename))[0]}.py")
    subprocess.run([f'jupyter nbconvert --to script {filename} --stdout > {fpy}'], shell=True)
    subprocess.run([f'git add {fpy}'], shell=True)
    dir_html = os.path.dirname(filename)
    subprocess.run([f'mkdir {dir_html}'], shell=True)
    subprocess.run([f'git add {dir_html}'], shell=True)
    fhtml = os.path.join(dir_html,  f"{os.path.splitext(os.path.basename(filename))[0]}.html")
    subprocess.run([f'jupyter nbconvert --to html {filename} --stdout > {fhtml}'], shell=True)
    subprocess.run([f'git add {fhtml}'], shell=True)
    return return_value

if __name__ == '__main__':
    exit(main())
