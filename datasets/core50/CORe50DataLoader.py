import os.path
import pickle
import random

import numpy as np
import torch
from PIL import Image

from datasets.core50 import constants
from datasets.core50.constants import NEW_TO_OLD_NAMES
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.utils import bordering_resize


class CORe50DataLoader(object):
    def __init__(
        self,
        root: str,
        original_image_size: tuple[int, int],
        input_image_size: tuple[int, int],
        resize_procedure: ResizeProcedure = ResizeProcedure.NONE,
        image_channels: int = 3,
        scenario: str = "ni",
        # Any value lower than 1 will lead to loading the entire batch
        mini_batch_size: int = 0,
        batch: int = 0,
        start_run: int = 0,
        start_idx: int = 0,
        randomize_data_order: bool = False,
        # If False, use the default 50 classes; if True, use the 10 categories,
        # by mapping each 5 classes into the corresponding category
        # (one category contains all the 5 different objects for each of the 10 types).
        category_based_split: bool = False,
    ):
        # Check that a resize procedure is provided when needed
        if original_image_size != input_image_size:
            assert (
                resize_procedure != ResizeProcedure.NONE
            ), "Resizing procedure not provided."

        # Check that enlarging methods only required in compatible cases
        if (
            original_image_size[0] <= input_image_size[0]
            and original_image_size[1] <= input_image_size[1]
        ):
            assert resize_procedure in [
                ResizeProcedure.BORDER,
            ], "Resizing procedure not compatible. Required to pass to greater size, but shrinking method provided."

        # Check if proper channel number
        assert image_channels in [1, 3], "Number of image channels not allowed."

        # Check if proper scenario
        assert scenario in NEW_TO_OLD_NAMES.keys(), "Provided scenario not allowed."

        # Extract class variables
        self.root = os.path.abspath(root)
        self.original_image_size = original_image_size
        self.input_image_size = input_image_size
        self.resize_procedure = resize_procedure
        self.in_channels = image_channels
        self.scenario = scenario
        self.mini_batch_size = mini_batch_size
        self.batch = batch
        self.run = start_run
        self.idx = start_idx
        self.randomize_data_order = randomize_data_order
        self.category_based_split = category_based_split

        self.n_batch = constants.N_BATCH

        if category_based_split:
            self.class_names = constants.CORE50_CATEGORY_NAMES
        else:
            self.class_names = constants.CORE50_CLASS_NAMES

        # Load necessary files
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

        # Randomize data order if needed
        print("Randomizing data order...")
        self.idx_order = None
        self.do_randomization()

    def __iter__(self):
        return self

    def __next__(self):
        # Mark end of batch
        if self.idx == len(self.idx_order) - 1:
            raise StopIteration

        # Prepare the list of image paths
        img_paths = list()

        # Load entire batch scenario
        batch_len = len(self.lup[self.scenario][self.run][self.batch])
        if self.mini_batch_size < 1 or self.mini_batch_size > batch_len:
            print(
                "Mini batch size provided determines the loading of the entire batch."
            )
            self.mini_batch_size = len(self.lup[self.scenario][self.run][self.batch])
        # If not, remove a number of random images from the train set, such that we exactly fill all the mini-batches
        else:
            self.idx_order = random.sample(
                self.idx_order, len(self.idx_order) - batch_len % self.mini_batch_size
            )

        # Prepare output tensor
        y = np.zeros(self.mini_batch_size, dtype=np.int64)

        for mini_batch_element in range(self.mini_batch_size):
            # Load image path
            img_paths.append(
                os.path.join(
                    self.root,
                    "core50_"
                    + str(self.original_image_size[0])
                    + "x"
                    + str(self.original_image_size[1]),
                    self.paths[self.idx_order[self.idx]],
                )
            )

            # Load label
            y[mini_batch_element] = self.labels[self.scenario][self.run][self.batch][
                self.idx_order[self.idx]
            ]
            self.idx += 1

        # Load images
        x = self.get_batch_from_paths(img_paths).astype(np.uint8)

        # Resize image if needed
        # Case 1: larger target image, resize by bordering (equal neutral gray border on either side)
        if self.resize_procedure == ResizeProcedure.BORDER:
            x = bordering_resize(
                x,
                input_image_size=self.input_image_size,
                original_image_size=self.original_image_size,
            )

        # Use category split if required
        if self.category_based_split:
            y = np.array([int(y_i / 5) for y_i in y], dtype=np.int64)

        return x, torch.from_numpy(y).to(torch.long)

    def get_batch_from_paths(self, paths):
        # Prepare variable to store the images
        x = np.zeros(
            (
                len(paths),
                self.original_image_size[0],
                self.original_image_size[1],
                self.in_channels,
            ),
            dtype=np.uint8,
        )

        # Iterate through paths and load images
        for i, path in enumerate(paths):
            # Load as grayscale
            if self.in_channels == 1:
                x[i, :, :, 0] = np.array(Image.open(path).convert("L"))
            # Load normally
            else:
                x[i] = np.array(Image.open(path))

        return x

    # Function that randomizes data order
    def do_randomization(self):
        self.idx_order = range(len(self.lup[self.scenario][self.run][self.batch]))
        if self.randomize_data_order:
            self.idx_order = random.sample(self.idx_order, len(self.idx_order))
