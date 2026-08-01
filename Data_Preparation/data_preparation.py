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


def sample_noise_factors(noise_type, size):
    if noise_type == "EM":
        low, high = 10, 100
    else:
        low, high = 20, 200
    return np.random.randint(low=low, high=high, size=size) / 100


def Data_Preparation(noise_type="BW", noise_version=1):
    print("Preparing data")

    np.random.seed(seed=1234)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    with open("data/QTDatabase.pkl", "rb") as file:
        qtdb = pickle.load(file)

    with open("data/NoiseBWL.pkl", "rb") as file:
        bw_signals, em_signals, ma_signals = pickle.load(file)

    if noise_type == "BW":
        signals = np.array(bw_signals)
    elif noise_type == "EM":
        signals = np.array(em_signals)
    elif noise_type == "MA":
        signals = np.array(ma_signals)
    else:
        raise ValueError("noise_type must be one of ['BW', 'EM', 'MA']")

    midpoint = int(signals.shape[0] / 2)
    noise_channel1_a = signals[:midpoint, 0]
    noise_channel1_b = signals[midpoint:-1, 0]
    noise_channel2_a = signals[:midpoint, 1]
    noise_channel2_b = signals[midpoint:-1, 1]

    if noise_version == 1:
        noise_test = noise_channel2_b
        noise_train = noise_channel1_a
    elif noise_version == 2:
        noise_test = noise_channel1_b
        noise_train = noise_channel2_a
    else:
        raise ValueError("noise_version must be 1 or 2")

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

    sn_train = []
    noise_index = 0
    rnd_train = sample_noise_factors(noise_type, len(beats_train))

    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max = np.max(noise) - np.min(noise)
        alpha = rnd_train[i] / (noise_max / beat_max)
        sn_train.append(beats_train[i] + alpha * noise)
        noise_index += samples
        if noise_index > len(noise_train) - samples:
            noise_index = 0

    sn_test = []
    noise_index = 0
    rnd_test = sample_noise_factors(noise_type, len(beats_test))

    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max = np.max(noise) - np.min(noise)
        alpha = rnd_test[i] / (noise_max / beat_max)
        sn_test.append(beats_test[i] + alpha * noise)
        noise_index += samples
        if noise_index > len(noise_test) - samples:
            noise_index = 0

    X_train = np.expand_dims(np.array(sn_train), axis=2)
    y_train = np.expand_dims(np.array(beats_train), axis=2)
    X_test = np.expand_dims(np.array(sn_test), axis=2)
    y_test = np.expand_dims(np.array(beats_test), axis=2)
    dataset = [X_train, y_train, X_test, y_test]

    os.makedirs("dataset", exist_ok=True)
    save_record_boundaries(flag_train, train_record_names)
    save_path = f"dataset/dataset_{noise_type.lower()}_nv{noise_version}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)

    print(f"Dataset saved to {save_path}")
    return dataset
