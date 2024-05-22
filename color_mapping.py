import numpy as np
from scipy.spatial.distance import euclidean
import json
import matplotlib.pyplot as plt


def map_scale_to_temperatures():
    colors = [
        "#FF0D05",
        "#FD2605",
        "#FF4F00",
        "#F8690C",
        "#F47A0B",
        "#FE9C06",
        "#FDB800",
        "#FDB306",
        "#FEF10E",
        "#F5FE07",
        "#D5FD03",
        "#BBFC0B",
        "#A0FA14",
        "#7BFD05",
        "#5FFF17",
        "#4DFA3B",
        "#23FE53",
        "#01FE76",
        "#02FFA5",
        "#0AFAC6",
        "#02FCE8",
        "#02EEFA",
        "#06DBF8",
        "#0EC3F6",
        "#01ACFB",
        "#0384FE",
        "#0169F6",
        "#0148FB",
        "#002CFF",
        "#0503FD",
        "#3B02F6",
        "#9600FE",
        "#C500FE",
        "#DF0BF2"
    ]

    hex_colors = colors[::-1]

    temp_min = 600
    temp_max = 900

    num_colors = len(hex_colors)

    temperature_range = np.linspace(temp_min, temp_max, num_colors)

    color_temp_map = dict(zip(hex_colors, temperature_range))
    return color_temp_map


color_scale = map_scale_to_temperatures()


def hex_to_rgb(hex_color):
    """Convert hex to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def find_nearest_color_temperature(unknown_color):
    unknown_rgb = hex_to_rgb(unknown_color)
    min_distance = float('inf')
    closest_colors = []

    for hex_color, temp in color_scale.items():
        known_rgb = hex_to_rgb(hex_color)
        distance = euclidean(unknown_rgb, known_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_colors = [(hex_color, temp, distance)]
        elif distance == min_distance:
            closest_colors.append((hex_color, temp, distance))

    if len(closest_colors) > 1:
        return sum(temp for _, temp, _ in closest_colors) / len(closest_colors)
    else:
        return closest_colors[0][1]


unknown_colors = ["#FEC000",
                  "#D7FE02",
                  "#92FC00",
                  "#B3FC00",
                  "#21FE57",
                  "#3EFA55",
                  "#DCFC06",
                  "#FAFB03",
                  "#BEFA01",
                  "#B0FE00",
                  "#F7FE04",
                  "#FEA403",
                  "#FF4403",
                  "#F83E0B"]
estimated_temperatures = []
for un_col in unknown_colors:
    estimated_temp = find_nearest_color_temperature(un_col)
    print(f"The estimated temperature for {un_col} is {estimated_temp:.2f}")
    estimated_temperatures.append(estimated_temp)

print("Temperatures: ", estimated_temperatures)

with open('model_config.json', 'r') as file:
    config = json.load(file)

L = config["L"]
xs = np.linspace(-L/2, L/2, len(estimated_temperatures))

# normalizacja do wartości maksymalnej, żeby te współczynniki nie były za duże
# coefficients = np.polyfit(xs, estimated_temperatures, 9)
# print("Coefficients: ", coefficients)
# # >>> numpy.polyfit(x, y, 2) # The 2 signifies a polynomial of degree 2
# # array([  -1.04978546,  115.16698544,  236.16191491])
#
# values = np.polyval(coefficients, xs)
# plt.plot(xs, values, marker='o', linestyle='-')
# plt.xlabel('X values')
# plt.ylabel('X values')
# plt.title(' Values Plot')
# plt.grid(True)
# plt.show()

N_values = [3, 4, 5, 6, 8, 11]
plt.figure(figsize=(10, 6))

for N in N_values:
    coefficients = np.polyfit(xs, estimated_temperatures, N)
    print(f"Coefficients for N={N}: ", coefficients)

    values = np.polyval(coefficients, xs)
    plt.plot(xs*1e3, values, marker='o', linestyle='-', label=f'stopień={N}')

plt.xlabel('x [mm]')
plt.ylabel('Estymowana temperatura [°C]')
plt.title('Estymacja głębokości trawienia przy użyciu wielomaniów różnego stopnia')
plt.grid(True)
plt.legend()
plt.show()
