import time
import matplotlib.pyplot as plt
import re
import numpy as np

DATA_PATH = "data/all"
DATA_PATH_SERVER = ""

f = open(DATA_PATH)
start = time.time()
len_set = set()
num_of_paragraphs = 0
for i, string in enumerate(f, 1):
    if string.startswith("<doc") or string.startswith("</doc>") or not string.endswith(".\n") or string == "\n":
        continue
    else:
        len_set.add(len(string.split()))
        num_of_paragraphs += 1
end = time.time()
print(f"Took {end - start} seconds")

print(f"Max paragraph lenght is {max(len_set)}")
print(f"Min paragraph length is {min(len_set)}")
print(f"Number of paragraphs is {num_of_paragraphs}")

plt.hist(list(len_set))
plt.show()
f.close()


def ends_with_punctutation(string):
    return re.match('.*[?.:;!]$', string) is not None


def unwanted_line(string):
    return string.startswith("<doc") or string.startswith("</doc>") or string == "\n" or not ends_with_punctutation(
        string)


#  Bunu geri geldiginde yap
def preprocess_text(input_file, output_train_file, output_val_file):
    print()


def remove_end_line(string):
    if string.endswith("\n"):
        string = string[:-1]
    return string


def random_indices_for_val_set(num_of_paragraphs_in_text, num_of_paragraphs_for_val):
    return set(np.random.choice(num_of_paragraphs, num_of_paragraphs_for_val, replace=False))


start = time.time()

all_data = open(DATA_PATH)
train_data = open("data/train",'a')
val_data = open("data/val",'a')
val_data_indices = random_indices_for_val_set(num_of_paragraphs,600)

for i,string in enumerate(all_data):
  if unwanted_line(string):
    continue
  else:
    string = remove_end_line(string)
    if i in val_data_indices:
      val_data.write(string)
    else:
      train_data.write(string)


train_data.close()
val_data.close()

end = time.time()
print(end-start)