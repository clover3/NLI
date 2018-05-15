
import subprocess

for f in range(1550, 3000, 50):
    arg = "{}".format(f)
    subprocess.call(["/net/home/youngwookim/env3/bin/python",
                     "/mnt/scratch/youngwookim/NLI/models/main.py",
                     "--frange",
                     arg])

