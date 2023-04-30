from PIL import Image
import os.path, sys

# PATH
path = "C:\\Users\\HP\\Documents\\r_workspace\\Chem"
dirs = os.listdir(path)

path_1 = "C:\\Users\\HP\\Documents\\r_workspace\\Chem\\1"
dirs_1 = os.listdir(path_1)

# CENTER
def crop():
    for item in dirs_1:
        fullpath = os.path.join(path_1,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((415, 40, 960, 700))
            imCrop.save(f + 'Center.png', "PNG", quality=100)

crop()

# LEFT
def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((140, 40, 675, 700))
            imCrop.save(f + 'Left.png', "PNG", quality=100)

crop()

# RIGHT
def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((700, 40, 1235, 700))
            imCrop.save(f + 'Right.png', "PNG", quality=100)

crop()
