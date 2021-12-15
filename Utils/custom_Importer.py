import builtins
import os
import sys
import json
import atexit

moduleName = sys.modules['__main__'].__file__.split(sep="/")[-1]
max = 0
counts_path = os.path.abspath("utils/libcounts.json")
done = False

def getjson():
    f = open(counts_path, "r")
    ret = json.loads(f.read())
    f.close()
    return ret


max = 0
count = 0
max_counts = {}
try:
    max_counts = getjson()
    max = max_counts[moduleName]
except:
    f = open(counts_path, "w")
    f.write("{}")
    f.close()

__builtin_import__ = builtins.__import__  # store a reference to the built-in import


def __custom_import__(name, *args, **kwargs):
    global count
    ret = __builtin_import__(name, *args, **kwargs)
    if done == False:
        count = count + 1
        if max > 0:
            printUpdate()
    return ret


def printUpdate():
    print(f"\r{count}/{max}", end="")

def doneImports():
    print(f"\r{count}/{count}", end="")
    updateFile()
    global done
    done = True

def updateFile():
    global max_counts
    if count != max and not doneImports:
        print("updating libcounts.json")
        if moduleName in max_counts.keys():
            max_counts[moduleName] = count
        else:
            max_counts[moduleName] = count
        f = open(counts_path, "w")
        f.write(json.dumps(max_counts))
        f.close()


atexit.register(updateFile)
builtins.__import__ = __custom_import__  # override the built-in import with our method
