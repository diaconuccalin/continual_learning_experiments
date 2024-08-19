import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from datasets.core50 import constants
from datasets.core50.constants import NEW_TO_OLD_NAMES
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.utils import bordering_resize


class CORE50DataLoader(object):
    def __init__(
        self,
        root: str,
        original_image_size: tuple[int, int],
        input_image_size: tuple[int, int],
        resize_procedure: ResizeProcedure = ResizeProcedure.NONE,
        channels: int = 3,
        scenario: str = "ni",
        load_entire_batch: bool = False,
        start_batch: int = 0,
        start_run: int = 0,
        start_idx: int = 0,
        eval_mode: bool = False,
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
        assert channels in [1, 3], "Number of channels not allowed."

        # Check if proper scenario
        assert scenario in NEW_TO_OLD_NAMES.keys(), "Provided scenario not allowed."

        # Extract class variables
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
        self.eval_mode = eval_mode

        self.n_batch = constants.N_BATCH
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

        if not self.load_batch:
            self.idx_list = self.lup[self.scenario][self.run][self.batch]

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.n_batch[self.scenario] and not self.eval_mode:
            raise StopIteration

        img_paths = list()
        if self.load_batch:
            # Load ids of batch
            self.idx_list = self.lup[self.scenario][self.run][self.batch]

            # Load image paths
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

            # Load labels
            y = np.asarray(
                self.labels[self.scenario][self.run][self.batch], dtype=np.int64
            )
            self.batch += 1
        else:
            # If reached end of current batch, go to next one
            if self.idx == len(self.idx_list) - 1:
                self.batch += 1
                self.idx_list = self.lup[self.scenario][self.run][self.batch]
                self.idx = 0

            # Load image path
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

            # Load label
            y = np.asarray(
                [
                    self.labels[self.scenario][self.run][self.batch][self.idx],
                ],
                dtype=np.int64,
            )
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

        # Transform to one hot encoding
        if not self.eval_mode:
            y = nn.functional.one_hot(torch.from_numpy(y), len(self.class_names))
        else:
            y = torch.from_numpy(y)

        return x, y

    def get_batch_from_paths(self, paths):
        # Prepare variable to store the images
        x = np.zeros(
            (
                len(paths),
                self.original_image_size[0],
                self.original_image_size[1],
                self.channels,
            ),
            dtype=np.uint8,
        )

        # Iterate through paths and load images
        for i, path in enumerate(paths):
            # Load as grayscale
            if self.channels == 1:
                x[i, :, :, 0] = np.array(Image.open(path).convert("L"))
            # Load normally
            else:
                x[i] = np.array(Image.open(path))

        return x
