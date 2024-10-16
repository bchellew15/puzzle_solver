# fails if destination file names exist

import os

ext = ".jpeg"
dir = "Pieces_16_shark/"

filenames = os.listdir(dir)
filenames.sort()
# only look at images
filenames = [name for name in filenames if ext in name]
print(filenames)

# rename
for i, name in enumerate(filenames):
	os.rename(dir + name, dir + "Piece" + str(i+1) + ext)
