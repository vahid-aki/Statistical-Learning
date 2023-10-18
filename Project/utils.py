import numpy as np
import os
import librosa
import noisereduce as nr
import itertools
import matplotlib.pyplot as plt


def dataset(path):
    labels = []
    audios = []
    indexes = []
    # sizes   = []
    # rates = []

    for r, d, f in os.walk(path):  # root, dir, file
        for file in f:
            if ".wav" in file:
                labels.append(r[-1])
                indexes.append(file[:-4])
                x, sr = librosa.load(os.path.join(r, file), sr=None)
                # rates.append(sr)
                # sizes.append(len(x))
                audios.append(x)

    Y = np.asarray(labels)
    X = np.asarray(audios, dtype=object)
    I = np.asarray(indexes)
    return X, Y, I, sr


def max_length(input):
    max = 0
    for i in input:
        if len(i) > max:
            max = len(i)
    return max


def pre_processing_dataset(input_dataset, sr=8000):
    audio_data = []
    for audio in input_dataset:
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)

        clips = librosa.effects.split(reduced_noise, top_db=10)
        wav_data = []
        for c in clips:
            data = reduced_noise[c[0] : c[1]]
            wav_data.extend(data)
        audio_data.append(wav_data)
    return audio_data


def zero_padding_data(data, max_len):
    output_audio = []
    for au in data:
        padded_audio = np.zeros(max_len)
        padded_audio[0 : len(au)] = au
        output_audio.append(padded_audio)
    return output_audio


def confusion_matrix_plot(cm):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("confusion_matrix for train")
    axs[1].set_title("confusion_matrix for test")
    for ax, c in zip(axs.flatten(), cm):
        ax.imshow(c, cmap="Blues", alpha=0.5)
        ax.set_xlabel("Predicted outputs", color="k")
        ax.set_ylabel("Actual outputs", color="k")
        ax.xaxis.set(ticks=range(10)), ax.yaxis.set(ticks=range(10))
        ax.set_ylim(9.5, -0.5), ax.grid(False)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, c[i, j], ha="center", va="center", c="g")
        fig.tight_layout()
    plt.show()
    return


def classification_report_plot(
    classificationReport, title="Classification report", cmap="RdBu"
):
    classificationReport = classificationReport.replace("\n\n", "\n")
    classificationReport = classificationReport.replace(" / ", "/")
    lines = classificationReport.split("\n")

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1 : len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ["Precision", "Recall", "F1-score"]
    yticklabels = [
        "{0} ({1})".format(class_names[idx], sup) for idx, sup in enumerate(support)
    ]

    plt.figure(figsize=(6, 6))
    plt.imshow(plotMat, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(
            j,
            i,
            format(plotMat[i, j], ".2f"),
            horizontalalignment="center",
            color="white"
            if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh)
            else "black",
        )

    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.tight_layout()
    plt.show()
    for line in lines[(len(lines) - 4) :]:
        print(line)
    return
