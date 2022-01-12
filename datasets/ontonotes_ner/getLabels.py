import sys

dict_tags = {}
for line in sys.stdin:

    wordlist = line.rstrip().split("\t")
    if len(wordlist) <= 1:
        continue
    tag = wordlist[1]
    dict_tags[tag] = 1

print(dict_tags.keys())