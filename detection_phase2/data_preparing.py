import glob
from os.path import join, exists
from os import remove
import numpy as np
from PIL import Image
from attention_phase1.attention import find_tfl_lights


def padding(image, padding_size):
    height, width, depth = image.shape

    padded_image = np.zeros((padding_size * 2 + height, padding_size * 2 + width, depth), int)
    padded_image[padding_size: -padding_size, padding_size: -padding_size] = image
    
    return padded_image


def save_files(data_file, label_file, cropped_ims, label):

    with open(data_file, mode='ab') as data_obj:
        for cropped_im in cropped_ims:
            np.array(cropped_im, dtype=np.uint8).tofile(data_obj)

    with open(label_file, mode='ab') as label_obj:
        for _ in cropped_ims:
            label_obj.write(label.to_bytes(1, byteorder='big', signed=False))


def crop_image(image, coordinate, size, padded=True):
    x, y = coordinate[0], coordinate[1]
    if not padded:
        image = padding(image, size // 2)
    crop_result = image[x: x + size, y: y + size]
    return crop_result


def crop_tfl_not_tfl(image_url, label_url):
    image = np.array(Image.open(image_url))
    label = np.array(Image.open(label_url))
    y_red, x_red, y_green, x_green = find_tfl_lights(image)

    pixels_of_r_tfl = [(x, y) for x, y in zip(x_red, y_red) if label[x, y] == 19]
    pixels_not_of_r_tfl = [(x, y) for x, y in zip(x_red, y_red) if label[x, y] != 19]
    pixels_of_g_tfl = [(x, y) for x, y in zip(x_green, y_green) if label[x, y] == 19]
    pixels_not_of_g_tfl = [(x, y) for x, y in zip(x_green, y_green) if label[x, y] != 19]

    crop_size = 81
    image = padding(image, crop_size // 2)

    list_r_tfl = [crop_image(image, pixel, crop_size) for pixel in pixels_of_r_tfl]
    list_not_tfl = [crop_image(image, pixel, crop_size) for pixel in pixels_not_of_r_tfl]
    list_g_tfl = [crop_image(image, pixel, crop_size) for pixel in pixels_of_g_tfl]
    list_not_tfl += [crop_image(image, pixel, crop_size) for pixel in pixels_not_of_g_tfl]

    return list_r_tfl, list_g_tfl, list_not_tfl


def get_lists_of_crops(image_list, label_list):
    list_r_tfl, list_g_tfl, list_not_tfl = [], [], []

    for i in range(len(image_list)):
        result = crop_tfl_not_tfl(image_list[i], label_list[i])
        list_r_tfl += result[0]
        list_g_tfl += result[1]
        list_not_tfl += result[2]
    length = min(len(list_r_tfl) + len(list_g_tfl), len(list_not_tfl))

    return list_r_tfl[:length // 2], list_g_tfl[:length // 2], list_not_tfl[:length]


def load_data(image_dir, label_dir, type_):

    im_cities_dirs = glob.glob(join(image_dir + type_, '*'))
    lab_cities_dirs = glob.glob(join(label_dir + type_, '*'))
    image_list = []
    label_list = []
    
    for city in im_cities_dirs:
        image_list += glob.glob(join(city, '*_leftImg8bit.png'))

    for city in lab_cities_dirs:
        label_list += glob.glob(join(city, "*_labelIds.png"))

    return image_list, label_list


def remove_old_file(label_file):
    if exists(label_file):
        remove(label_file)


def prepare_data(root_dir, data_dir, type_):
    image_list, label_list = load_data(join(root_dir, "leftImg8bit_trainvaltest/leftImg8bit/"),
                                       join(root_dir, "gtFine_trainvaltest/gtFine/"), type_)
    data_file = join(data_dir, type_, 'data.bin')
    label_file = join(data_dir, type_, 'labels.bin')

    remove_old_file(data_file)
    remove_old_file(label_file)

    level_size = 100  # I defined this size in order to maximize the data without getting a memory error

    while len(image_list) > level_size:
        cropped_images = get_lists_of_crops(image_list[:level_size], label_list[:level_size])
        list_r_tfl, list_g_tfl, list_not_tfl = cropped_images
        is_tfl, is_not_tfl = 1, 0
        save_files(data_file, label_file, list_r_tfl, is_tfl)
        save_files(data_file, label_file, list_g_tfl, is_tfl)
        save_files(data_file, label_file, list_not_tfl, is_not_tfl)
        image_list, label_list = image_list[level_size:], label_list[level_size:]


def main():
    root_dir = "../../data/"
    data_dir = join(root_dir, "dataset/")

    prepare_data(root_dir, data_dir, "train/")
    prepare_data(root_dir, data_dir, "val/")


if __name__ == '__main__':
    main()
