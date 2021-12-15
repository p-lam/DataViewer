import json
import pandas as pd
from datetime import datetime


def __EOH_BioSignal(path):
    tmp = open(path)
    eoh = 0
    for line in tmp.readlines():
        eoh += 1
        if line.find("# EndOfHeader") >= 0:
            return eoh
    return 0

def __EOH_DSI(path):
    tmp = open(path)
    eoh = 0
    for line in tmp.readlines():
        if line.find("#") < 0:
            return eoh
        eoh += 1
    return 0

def readDSI(path):
    print(f"Loading data from {path}")
    eoh = __EOH_DSI(path)
    data = pd.read_csv(path, skiprows=eoh)
    fs = 300
    print(f"\tSampling rate {fs}hz")
    print(data.head())
    return data, fs

def readBioSignals(path):
    print(f"Loading data from {path}")
    eoh = __EOH_BioSignal(path)
    eoh += 10  # skips first 10 lines
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[0, 2, 3, 4],
                       names=["Time", "EEG", "fNIRS1", "fNIRS2"])
    try:
        file1 = open(path, 'r')
        jsonText = file1.readlines()[1][2:]
        headerJson = json.loads(jsonText)
        headerJson = headerJson[list(headerJson)[0]]
        fs = headerJson["sampling rate"]
        file1.close()
    except KeyError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    except json.decoder.JSONDecodeError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    print(f"Sampling rate {fs}hz")
    # print(data.head())

    return data, fs, t0


def readSustainedFile(path, t0, sr):
    print(f"Loading data from {path}")
    data = pd.read_csv(path, skiprows=1, header=None, usecols=[0, 18, 19, 20],
                       names=["Time", "Input", "Correct", "Reaction"])
    # data = data[data["Reaction"] != -1]
    toTime = lambda s: ((datetime.strptime("2000:" + s, "%Y:%H:%M:%S.%f").timestamp() * 1000 - t0) / 1000)
    data["Time"] = pd.Series([toTime(s) for s in data["Time"]])
    return data[data["Reaction"] != -1]