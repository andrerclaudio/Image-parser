import logging
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Log parameters adjustment call
# _ Hour (24 hours format)
# _ Minutes
# _ Seconds
# _ Month-Day
# _ Level to print and above
# _ Message to show

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] - %(levelname)s:  %(message)s',
                    datefmt='%H:%M:%S %m-%d-%y')


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image():
    bgr_image = cv2.imread('/home/andre/Desktop/atom2.png')
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def elbow_method():
    # calculate distortion for a range of number of cluster
    distortions = []
    for i in range(1, number_of_clusters_check):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(modified_image)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, number_of_clusters_check), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def color_predominance_identification(image, n):
    clf = KMeans(n_clusters=n)
    labels = clf.fit_predict(image)

    counts = Counter(labels)
    center_colors = clf.cluster_centers_

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    plt.figure(2, figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.title('KMeans colors')
    plt.savefig('flowers_colors.png', transparent=True)
    plt.show()


def most_color_occurrences(image):
    hex_colors_values = []
    color_rgb = []
    df = pd.DataFrame.from_records(image, columns=('R', 'G', 'B'))

    for idx in df.index:
        color_rgb.append(df['R'][idx])
        color_rgb.append(df['G'][idx])
        color_rgb.append(df['B'][idx])
        hex_colors_values.append(RGB2HEX(color_rgb))
        color_rgb = []

    c_pixels = Counter(hex_colors_values)
    a = c_pixels.most_common(20)
    a_listed = []
    pixels_listed = []

    for i in a:
        v, qtd = i
        a_listed.append(qtd)
        pixels_listed.append(v)

    plt.figure(1, figsize=(8, 6))
    plt.pie(a_listed, labels=pixels_listed, colors=pixels_listed)
    plt.title('Counted occurrences for a specific color')
    plt.show()


def application():
    """" All application has its initialization from here """
    logging.info('Main application is running!')
    
    n = 11
    
    image = get_image()

    # modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    # elbow_method()
    # number_of_clusters_query = int(input('Number of clusters:   '))
    color_predominance_identification(modified_image, n)
    most_color_occurrences(modified_image)
