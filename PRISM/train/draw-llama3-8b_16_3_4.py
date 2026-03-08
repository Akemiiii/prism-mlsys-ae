import numpy as np
from matplotlib import pyplot as plt

config = {
    "EAGLE2": {
        "color": "#8c564b",
        "marker": "o"
    },
    "HASS": {
        "color": "#1f77b4",
        "marker": "v"
    },
    "EAGLE3": {
        "color": "#2ca02c",
        "marker": "d"
    },
    "PRISM": {
        "color": "#d62728",
        "marker": "P"
    }
}

xticks = ["100k", "200k", "400k", "600k", "800k"]

means = [
    (
        "EAGLE2",
        [[3.53790, 3.61489, 3.67235, 3.70120, 3.69206],
         [3.36394, 3.44067, 3.48443, 3.53196, 3.51106]]
    ),
    (
        "HASS",
        [[3.70512, 3.78453, 3.85574, 3.86508, 3.85948],
         [3.52721, 3.59580, 3.65260, 3.66546, 3.65139]]
    ),
    (
        "EAGLE3",
        [[3.77928, 3.92244, 4.02642, 4.08573, 4.07757],
         [3.55682, 3.68045, 3.80578, 3.84308, 3.84926]]
    ),
    (
        "PRISM",
        [[3.99624, 4.06834, 4.18146, 4.21871, 4.23785],
         [3.75506, 3.83250, 3.92500, 3.95642, 3.97479]]
    )
]

for temperature in [0, 1]:
    plt.figure(figsize=(7, 7))

    legends = []
    for model, values in means:
        plt.plot(
            range(1, 6),
            values[temperature],
            color=config[model]["color"],
            marker=config[model]["marker"],
            markersize=9,
            linestyle="-" if temperature == 0 else ":"
        )
        legends.append(model)

    plt.grid()
    plt.legend(legends, loc="lower right", fontsize=12)
    plt.title(f"Temperature = {temperature}", fontsize=18)

    plt.xticks(range(1, 6), xticks, fontsize=12)
    plt.xlabel("Train Data Volume", fontsize=15)

    if temperature == 0:
        plt.yticks(np.arange(3.50, 4.31, 0.1), fontsize=12)
    else:
        plt.yticks(np.arange(3.30, 4.11, 0.1), fontsize=12)
    plt.ylabel("Acceptance Length", fontsize=18)
