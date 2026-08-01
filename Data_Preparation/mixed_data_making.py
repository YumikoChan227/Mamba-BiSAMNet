import os
import _pickle as pickle

import numpy as np

from Data_Preparation import Prepare_NSTDB, Prepare_QTDatabase


def save_record_boundaries(flag_train, train_record_names):
    flag_train = np.asarray(flag_train, dtype=np.int64)
    train_record_names = np.asarray(train_record_names)
    flag_path = "dataset/flag_train.npy"
    names_path = "dataset/train_record_names.npy"

    if os.path.exists(flag_path):
        saved_flag = np.load(flag_path).astype(np.int64).ravel()
        if not np.array_equal(saved_flag, flag_train):
            raise RuntimeError(
                "Generated record boundaries differ from dataset/flag_train.npy. "
                "Regenerate all dataset variants together."
            )

    if os.path.exists(names_path):
        saved_names = np.load(names_path)
        if not np.array_equal(saved_names, train_record_names):
            raise RuntimeError(
                "Generated record order differs from "
                "dataset/train_record_names.npy. Regenerate all dataset variants together."
            )

    np.save(flag_path, flag_train)
    np.save(names_path, train_record_names)


def Data_Preparation(noise_version=1):
    print("Preparing mixed-noise data")

    np.random.seed(seed=1234)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    with open("data/QTDatabase.pkl", "rb") as file:
        qtdb = pickle.load(file)

    with open("data/NoiseBWL.pkl", "rb") as file:
        bw_signals, em_signals, ma_signals = pickle.load(file)

    def split_noise(signal):
        midpoint = int(len(signal) / 2)
        channel1_a = signal[:midpoint, 0]
        channel1_b = signal[midpoint:-1, 0]
        channel2_a = signal[:midpoint, 1]
        channel2_b = signal[midpoint:-1, 1]

        if noise_version == 1:
            return channel1_a, channel2_b
        if noise_version == 2:
            return channel2_a, channel1_b
        raise ValueError("noise_version must be 1 or 2")

    bw_train, bw_test = split_noise(bw_signals)
    em_train, em_test = split_noise(em_signals)
    ma_train, ma_test = split_noise(ma_signals)

    beats_train = []
    beats_test = []
    test_set = [
        "sel123",
        "sel233",
        "sel302",
        "sel307",
        "sel820",
        "sel853",
        "sel16420",
        "sel16795",
        "sele0106",
        "sele0121",
        "sel32",
        "sel49",
        "sel14046",
        "sel15814",
    ]
    samples = 512
    flag_train = []
    train_record_names = []

    for signal_name in qtdb:
        is_test = signal_name in test_set
        record_start = len(beats_train)

        for beat in qtdb[signal_name]:
            padded_beat = np.zeros(samples)
            beat = np.array(beat)
            init_padding = 16

            if beat.shape[0] > samples - init_padding:
                continue

            padded_beat[init_padding:beat.shape[0] + init_padding] = (
                beat - (beat[0] + beat[-1]) / 2
            )

            if is_test:
                beats_test.append(padded_beat)
            else:
                beats_train.append(padded_beat)

        if not is_test and len(beats_train) > record_start:
            flag_train.append(record_start)
            train_record_names.append(signal_name)

    def add_mixed_noise(beats, bw_noise, em_noise, ma_noise):
        noisy_beats = []
        noise_index = 0
        weights = np.random.uniform(low=0.0, high=0.6, size=(len(beats), 3))

        for i in range(len(beats)):
            bw = bw_noise[noise_index:noise_index + samples]
            em = em_noise[noise_index:noise_index + samples]
            ma = ma_noise[noise_index:noise_index + samples]

            beat_max = np.max(beats[i]) - np.min(beats[i])
            scales = np.array(
                [
                    (np.max(bw) - np.min(bw)) / beat_max,
                    (np.max(em) - np.min(em)) / beat_max,
                    (np.max(ma) - np.min(ma)) / beat_max,
                ]
            )
            alpha_bw, alpha_em, alpha_ma = weights[i] / scales
            noisy_beats.append(
                beats[i] + alpha_bw * bw + alpha_em * em + alpha_ma * ma
            )

            noise_index += samples
            if noise_index > len(bw_noise) - samples:
                noise_index = 0

        return np.array(noisy_beats)

    sn_train = add_mixed_noise(beats_train, bw_train, em_train, ma_train)
    sn_test = add_mixed_noise(beats_test, bw_test, em_test, ma_test)

    X_train = np.expand_dims(sn_train, axis=2)
    y_train = np.expand_dims(np.array(beats_train), axis=2)
    X_test = np.expand_dims(sn_test, axis=2)
    y_test = np.expand_dims(np.array(beats_test), axis=2)
    dataset = [X_train, y_train, X_test, y_test]

    os.makedirs("dataset", exist_ok=True)
    save_record_boundaries(flag_train, train_record_names)
    save_path = f"dataset/dataset_mixed_nv{noise_version}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)

    print(f"Mixed-noise dataset saved to {save_path}")
    return dataset
