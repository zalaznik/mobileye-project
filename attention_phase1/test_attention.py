import os
import json
import glob
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from attention import find_tfl_lights


def show_image_and_gt(image: np.ndarray, objs, fig_num=None) -> None:
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path: str, json_path=None, fig_num=None) -> None:
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=2)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=2)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../data/leftImg8bit_trainvaltest/leftImg8bit/train/bremen'
    if args.dir is None:
        args.dir = default_base
    f_list = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))[70:78]

    for image in f_list:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(f_list):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
