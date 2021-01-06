import matplotlib.pyplot as plt


class FrameContainer(object):
    def __init__(self, img_path: str) -> None:
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
