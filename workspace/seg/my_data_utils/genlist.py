import os
import sys
import random

directory = sys.argv[1]
trainlist = open("train.txt", "w")
vallist = open("val.txt", "w")

for filename in os.listdir(directory):
  if filename.endswith(".jpg"):
    filename_base = os.path.splitext(os.path.split(filename)[1])[0]
    x = random.random()
    if x < 0.03:
      vallist.write(filename_base + '\n')
    else:
      trainlist.write(filename_base + '\n')

trainlist.close()
vallist.close()

