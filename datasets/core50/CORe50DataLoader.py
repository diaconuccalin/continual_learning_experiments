import os.path
import pickle
import random

import numpy as np
import torch
from PIL import Image

from datasets.core50 import constants
from datasets.core50.constants import NEW_TO_OLD_NAMES
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.vit_lr_utils import bordering_resize


class CORe50DataLoader(object):
    def __init__(
        self,
        root: str,
        original_image_size: tuple[int, int],
        input_image_size: tuple[int, int],
        resize_procedure: ResizeProcedure = ResizeProcedure.NONE,
        image_channels: int = 3,
        scenario: str = "ni",
        exemplar_set_ratio: float = 0.0,
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
        debug_mode: bool = False,
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
        assert scenario in NEW_TO_OLD_NAMES.values(), "Provided scenario not allowed."

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
        self.debug_mode = debug_mode

        # Rehearsal variables
        self.exemplar_set_ratio = exemplar_set_ratio
        self.exemplar_set = dict()
        self.current_set = dict()

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

        # print("Loading labels...")
        # with open(os.path.join(self.root, "labels.pkl"), "rb") as f:
        #     self.labels = pickle.load(f)
        # print("Labels loaded")

        # Randomize data order if needed
        print("Randomizing data order if required...")
        self.idx_order = None
        self.do_randomization()

    def __iter__(self):
        return self

    def __next__(self):
        # Mark end of batch
        if self.idx == len(self.idx_order):
            raise StopIteration

        # Prepare the list of image paths
        img_paths = list()

        # Prepare output tensor
        y = np.zeros(self.mini_batch_size, dtype=np.int64)

        for mini_batch_element in range(self.mini_batch_size):
            # Get file path
            file_path = self.paths[self.idx_order[self.idx]]

            # Load image path
            img_paths.append(
                os.path.join(
                    self.root,
                    "core50_"
                    + str(self.original_image_size[0])
                    + "x"
                    + str(self.original_image_size[1]),
                    file_path,
                )
            )

            # Determine label
            y[mini_batch_element] = (
                int(file_path.split("/o")[1].split("/C")[0].split("/")[0]) - 1
            )

            # Add classes to encountered dict
            if y[mini_batch_element] not in self.current_set.keys():
                self.current_set[y[mini_batch_element]] = [self.idx_order[self.idx]]
            else:
                self.current_set[y[mini_batch_element]].append(self.idx_order[self.idx])

            self.idx += 1

        # Don't load images in debug mode
        if self.debug_mode:
            return None, None

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
        # Get current batch ids
        if isinstance(self.batch, list):
            self.idx_order = list()
            for b in self.batch:
                self.idx_order += list(self.lup[self.scenario][self.run][b])
        else:
            self.idx_order = list(self.lup[self.scenario][self.run][self.batch])

        # Add exemplar set to id list
        for key in self.exemplar_set.keys():
            self.idx_order += self.exemplar_set[key]

        # Randomize their order if required
        if self.randomize_data_order:
            self.idx_order = random.sample(self.idx_order, len(self.idx_order))

        # Load entire batch scenario
        batch_len = len(self.idx_order)
        if self.mini_batch_size < 1 or self.mini_batch_size > batch_len:
            print(
                "Mini batch size provided determines the loading of the entire batch."
            )
            self.mini_batch_size = batch_len
        # If not, remove a number of random images from the train set, such that we exactly fill all the mini-batches
        elif self.mini_batch_size != 1:
            self.idx_order = random.sample(
                self.idx_order,
                len(self.idx_order) - batch_len % self.mini_batch_size,
            )

        return None

    def update_batch(self, batch):
        # Function to do necessary operations when changing batch
        self.batch = batch
        self.idx = 0
        self.do_randomization()

        return None

    def update_exemplar_set(self):
        # Check current stored exemplar set size
        total_available = 0
        for key in self.exemplar_set.keys():
            total_available += len(self.exemplar_set[key])
        for key in self.current_set.keys():
            total_available += len(self.current_set[key])

        if self.exemplar_set_ratio >= 1.0 or total_available <= (
            self.exemplar_set_ratio * constants.NC_TRAINING_SET_SIZE
        ):
            # If enough space left, add all samples from the current batch
            for key in self.current_set.keys():
                self.exemplar_set[key] = self.current_set[key]
        else:
            # Else, keep equal number of samples for each class
            # Count all available samples (stored exemplars + current batch)
            keys_so_far = list()
            keys_so_far += list(self.exemplar_set.keys())
            keys_so_far += list(self.current_set.keys())

            # Compute how many samples to store per class
            samples_per_class = int(
                (self.exemplar_set_ratio * constants.NC_TRAINING_SET_SIZE)
                / len(set(keys_so_far))
            )

            # Keep the predetermined number of samples from previous exemplars
            for key in self.exemplar_set.keys():
                if len(self.exemplar_set[key]) < samples_per_class:
                    self.exemplar_set[key] = self.exemplar_set[key]
                else:
                    self.exemplar_set[key] = random.sample(
                        self.exemplar_set[key], samples_per_class
                    )

            # Keep the predetermined number of samples from current batch
            for key in self.current_set.keys():
                if len(self.current_set[key]) < samples_per_class:
                    self.exemplar_set[key] = self.current_set[key]
                else:
                    self.exemplar_set[key] = random.sample(
                        self.current_set[key], samples_per_class
                    )

        # Reset dictionary of current batch
        self.current_set = dict()

        # In debug mode, print the number of exemplars stored for each class
        if self.debug_mode:
            total_number_of_samples = 0

            the_keys = list(self.exemplar_set.keys())
            the_keys.sort()

            print("Total number of keys:", len(the_keys))
            for el in the_keys:
                print(el, "-", len(self.exemplar_set[el]))
                total_number_of_samples += len(self.exemplar_set[el])
            print("Total number of stored exemplars:", total_number_of_samples)
            print(
                "Required maximum number of exemplars:",
                int(self.exemplar_set_ratio * constants.NC_TRAINING_SET_SIZE),
            )

        return None
