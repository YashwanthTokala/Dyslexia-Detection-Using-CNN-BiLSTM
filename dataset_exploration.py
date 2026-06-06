import os

for root, dirs, files in os.walk("/kaggle/input", topdown=True):
    for f in files:
        print(os.path.join(root, f))