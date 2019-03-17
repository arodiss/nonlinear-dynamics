import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
import numpy as np

IMAGE_SIZE = 1200


def embed(series, tau, dims):
    embeddings = []
    for index in range(0, len(series)):
        try:
            new_embedding = []
            for d in range(0, dims):
                new_embedding.append(series[index + d * tau])
            embeddings.append(new_embedding)
        except IndexError:
            pass
    return embeddings


def explore_tau(series, dims=7, min_tau=1, max_tau=350):
    try:
        os.remove('output.mp4')
    except OSError:
        pass

    video_writer = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    for tau in tqdm(range(min_tau, max_tau), total=max_tau-min_tau):
        plt.figure(num=None, figsize=(20, 20), dpi=int(IMAGE_SIZE/20), facecolor='w', edgecolor='k')
        embeddings = embed(series, tau, dims)
        plt.plot(
            [e[0] for e in embeddings],
            [e[1] for e in embeddings],
        )
        plt.title('Tau = {}'.format(tau), fontsize=30)
        plt.savefig('figure.png')
        plt.close()
        video_writer.write(cv2.resize(cv2.imread('figure.png'), (IMAGE_SIZE, IMAGE_SIZE)))
        os.remove('figure.png')
    video_writer.release()


def plot_mutual_info(series, min_tau=1, max_tau=350, bins=100):
    taus = []
    mutual_infos = []
    for tau in tqdm(range(min_tau, max_tau), total=max_tau-min_tau):
        taus.append(tau)
        contingency = np.histogram2d(series[0:-tau], series[tau:], bins)[0]
        mutual_infos.append(mutual_info_score(None, None, contingency=contingency))
    plt.xlabel('Tau')
    plt.ylabel('Mutual info')
    plt.plot(taus, mutual_infos)
    plt.savefig('mutual-info.png')
    plt.close()


if __name__ == "__main__":
    series = list(pd.read_csv('amplitude.dat', header=None)[0])
    explore_tau(series)

