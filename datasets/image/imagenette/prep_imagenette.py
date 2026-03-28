from os import listdir, makedirs
from shutil import copy2

makedirs("./test")
makedirs("./train")

test_files = sorted(f for f in listdir("./test_all") if f.endswith("jpg"))
train_files = sorted(f for f in listdir("./train_all") if f.endswith("jpg"))

############
test_count_bylabel = {}

for f in test_files:
  label = f.split("_")[0]
  if label not in test_count_bylabel: test_count_bylabel[label] = 0
  elif test_count_bylabel[label] >= 150: continue
  test_count_bylabel[label] += 1
  copy2("./test_all/"+f, "./test/"+f)


##########
train_count_bylabel = {}

for f in train_files:
  label = f.split("_")[0]
  if label not in train_count_bylabel: train_count_bylabel[label] = 0
  elif train_count_bylabel[label] >= 300: continue
  train_count_bylabel[label] += 1
  copy2("./train_all/"+f, "./train/"+f)
