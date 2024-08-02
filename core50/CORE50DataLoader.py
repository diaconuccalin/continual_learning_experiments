import os.path
import pickle

import numpy as np
from PIL import Image

from core50 import constants


class CORE50DataLoader(object):
    def __init__(
        self,
        root: str,
        original_image_size: tuple[int, int],
        input_image_size: tuple[int, int],
        resize_procedure: str = "",
        channels: int = 3,
        scenario: str = "ni",
        load_entire_batch: bool = False,
        start_batch: int = 0,
        start_run: int = 0,
        start_idx: int = 0,
    ):
        self.root = os.path.abspath(root)
        self.original_image_size = original_image_size
        self.input_image_size = input_image_size
        self.resize_procedure = resize_procedure
        self.channels = channels
        self.scenario = scenario
        self.load_batch = load_entire_batch
        self.batch = start_batch
        self.run = start_run
        self.idx = start_idx

        self.nbatch = constants.nbatch
        self.class_names = constants.classnames

        print("Loading paths...")
        with open(os.path.join(self.root, "paths.pkl"), "rb") as f:
            self.paths = pickle.load(f)
        print("Paths loaded")

        print("Loading LUP...")
        with open(os.path.join(self.root, "LUP.pkl"), "rb") as f:
            self.lup = pickle.load(f)
        print("LUP loaded")

        print("Loading labels...")
        with open(os.path.join(self.root, "labels.pkl"), "rb") as f:
            self.labels = pickle.load(f)
        print("Labels loaded")

        if not self.load_batch:
            self.idx_list = self.lup[self.scenario][self.run][self.batch]

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.nbatch[self.scenario]:
            raise StopIteration

        img_paths = list()
        if self.load_batch:
            self.idx_list = self.lup[self.scenario][self.run][self.batch]
            for current_idx in self.idx_list:
                img_paths.append(
                    os.path.join(
                        self.root,
                        "core50_"
                        + str(self.original_image_size[0])
                        + "x"
                        + str(self.original_image_size[1]),
                        self.paths[current_idx],
                    )
                )
            y = np.asarray(
                self.labels[self.scenario][self.run][self.batch], dtype=np.uint8
            )
            self.batch += 1
        else:
            if self.idx == len(self.idx_list) - 1:
                self.batch += 1
                self.idx_list = self.lup[self.scenario][self.run][self.batch]
                self.idx = 0

            img_paths.append(
                os.path.join(
                    self.root,
                    "core50_"
                    + str(self.original_image_size[0])
                    + "x"
                    + str(self.original_image_size[1]),
                    self.paths[self.idx_list[self.idx]],
                )
            )
            y = np.asarray(
                [
                    self.labels[self.scenario][self.run][self.batch][self.idx],
                ],
                dtype=np.uint8,
            )
            self.idx += 1

        x = self.get_batch_from_paths(img_paths).astype(np.uint8)

        if self.resize_procedure == "border":
            new_x = np.full(
                (
                    x.shape[0],
                    self.input_image_size[0],
                    self.input_image_size[1],
                    x.shape[-1],
                ),
                fill_value=128,
            )

            border_size_0 = int(
                (self.input_image_size[0] - self.original_image_size[0]) / 2
            )
            border_size_1 = int(
                (self.input_image_size[1] - self.original_image_size[1]) / 2
            )

            new_x[:, border_size_0:-border_size_0, border_size_1:-border_size_1, :] = x
            x = new_x

        return x, y

    def get_batch_from_paths(self, paths):
        x = np.zeros(
            (
                len(paths),
                self.original_image_size[0],
                self.original_image_size[1],
                self.channels,
            ),
            dtype=np.uint8,
        )

        for i, path in enumerate(paths):
            if self.channels == 1:
                x[i, :, :, 0] = np.array(Image.open(path).convert("L"))
            else:
                x[i] = np.array(Image.open(path))

        return x
