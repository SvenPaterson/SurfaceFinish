import numpy as np
import matplotlib.pyplot as plt


def sine_wave(x, frequency, amplitude, phase):
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
    return y



def main():
    SAMPLE_RATE = 0.001
    FREQ = 0.8

    x = np.linspace(0, 5.6, int(1/SAMPLE_RATE))
    signal = sine_wave(x, FREQ, 1, 0)
    filter = gaussian_distribution(FREQ, SAMPLE_RATE)

    # do 1st plotting here
    plt.subplot(211)
    plt.plot(x, filter)
    plt.axis([-1, 1, 0, 1])
    plt.title('Gaussian Filter')

    # do 2nd plotting here
    plt.subplot(212)
    plt.plot(x, signal, 'b')
    plt.axis([0, 5.6, -1.5, 1.5])
    plt.title('Signal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()