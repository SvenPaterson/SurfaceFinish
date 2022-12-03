import matplotlib.pyplot as plt
import numpy as np


def generate_sine_wave(freq, sample_rate, length):
    x = np.linspace(0, length, sample_rate * length, endpoint = False)
    frequencies = x * freq
    y = np.sin(frequencies * 2 * np.pi)
    return x, y


def main():
    SAMPLE_RATE = 4000
    LENGTH = 5
    x, tone_a = generate_sine_wave(1, SAMPLE_RATE, LENGTH)
    _, noise_tone = generate_sine_wave(15, SAMPLE_RATE, LENGTH)
    mixed_tone = tone_a + noise_tone * 0.3
    plt.plot(x, mixed_tone)
    plt.show()


if __name__ == "__main__":
    main()