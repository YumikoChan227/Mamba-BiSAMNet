import os
import _pickle as pickle

import wfdb


def prepare(NSTDBPath="data/mit-bih-noise-stress-test-database-1.0.0/"):
    bw_signals, _ = wfdb.rdsamp(NSTDBPath + "bw")
    em_signals, _ = wfdb.rdsamp(NSTDBPath + "em")
    ma_signals, _ = wfdb.rdsamp(NSTDBPath + "ma")

    os.makedirs("data", exist_ok=True)
    with open("data/NoiseBWL.pkl", "wb") as output:
        pickle.dump([bw_signals, em_signals, ma_signals], output)

    print("NSTDB saved to data/NoiseBWL.pkl")
