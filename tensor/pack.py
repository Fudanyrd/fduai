#!/usr/bin/python3
import subprocess
import os

if __name__ == "__main__":
    cc = [f for f in os.listdir('.') if f.endswith('.cc') or f.endswith('.cpp')]
    cu = [f for f in os.listdir('.') if f.endswith('.cu')]
    h = [f for f in os.listdir('.') if f.endswith('.h')]
    py = [f for f in os.listdir('.') if f.endswith('.py')]

    subprocess.run(
        ['tar', '-czvf', 'libtensor.tar.gz', 'cc', 'readme.txt', 'requirements.txt'] + cc + cu + h + py,
        check=True
    )
