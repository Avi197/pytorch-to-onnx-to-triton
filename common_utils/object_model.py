from PIL import Image


class ResultObject:
    def __init__(self, coord, img):
        self.coord = coord
        self.img = img
        self.label = None

    def get_all_x(self):
        all_x = [int(coord[0]) for coord in self.coord]
        min_x = min(all_x)
        max_x = max(all_x)
        return all_x, min_x, max_x

    def get_all_y(self):
        all_y = [int(coord[1]) for coord in self.coord]
        min_y = min(all_y)
        max_y = max(all_y)
        return all_y, min_y, max_y

    def get_coord(self):
        return self.coord

    def set_coord(self, coord):
        self.coord = coord

    def get_img(self):
        return self.img

    def get_img_l(self):
        return Image.fromarray(self.img, 'L')

    def set_img(self, img):
        self.img = img

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label
