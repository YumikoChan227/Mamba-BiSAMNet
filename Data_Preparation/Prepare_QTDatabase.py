import glob
import math
import os
import _pickle as pickle

import numpy as np
import wfdb
from scipy.signal import resample_poly


def prepare(QTpath="data/qt-database-1.0.0/"):
    newFs = 360
    namesPath = sorted(glob.glob(QTpath + "/*.dat"))
    QTDatabaseSignals = {}

    for path in namesPath:
        record_path = path.rsplit(".dat", 1)[0]
        register_name = os.path.basename(record_path)
        signal, fields = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, "pu1")

        Anntype = np.array(ann.symbol)
        annSamples = ann.sample
        Pidx = annSamples[Anntype == "p"]
        Sidx = annSamples[Anntype == "("]
        Ridx = annSamples[Anntype == "N"]

        ind = np.zeros(len(Pidx))
        for j in range(len(Pidx)):
            arr = np.where(Pidx[j] > Sidx)[0]
            ind[j] = arr[-1]

        Pstart = Sidx[ind.astype(np.int64)]
        Pstart = Pstart - int(0.04 * fields["fs"])
        auxSig = signal[:, 0]

        beats = []
        for k in range(len(Pstart) - 1):
            remove = (Ridx > Pstart[k]) & (Ridx < Pstart[k + 1])
            if np.sum(remove) < 2:
                beats.append(auxSig[Pstart[k]:Pstart[k + 1]])

        beatsRe = []
        for beat in beats:
            length = math.ceil(len(beat) * newFs / fields["fs"])
            normBeat = list(reversed(beat)) + list(beat) + list(reversed(beat))
            resampled = resample_poly(normBeat, newFs, fields["fs"])
            beatsRe.append(resampled[length - 1:2 * length - 1])

        QTDatabaseSignals[register_name] = beatsRe

    os.makedirs("data", exist_ok=True)
    with open("data/QTDatabase.pkl", "wb") as output:
        pickle.dump(QTDatabaseSignals, output)

    print("QT Database saved to data/QTDatabase.pkl")
