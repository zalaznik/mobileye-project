from os.path import join
import numpy as np
import matplotlib.pyplot as plt


def read_files(data_dir, index):
    data_file = join(data_dir, 'data.bin')
    label_file = join(data_dir, 'labels.bin')
    crop_size = 81
    data = np.memmap(data_file, dtype=np.uint8, mode='r', shape=(crop_size, crop_size, 3),
                     offset=crop_size * crop_size * 3 * index)
    label = np.memmap(label_file, dtype=np.uint8, mode='r', shape=(1,), offset=index)
    plt.imshow(data)
    plt.title(f"Traffic light" if label else f"Not Traffic light")
    plt.show(block=True)


def main():
    data_dir = "../../data/dataset/"

    for index in range(0, 4000, 100):
        read_files(join(data_dir, "train/"), index)


if __name__ == '__main__':
    main()
