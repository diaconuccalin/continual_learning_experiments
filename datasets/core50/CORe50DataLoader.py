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
        rehearsal_memory_size: int = 0,
        # Not using actual mini-batches larger than 1 anymore, but working with "virtual" mini-batches
        # (One sample fed at a time and model updated after a number of samples equivalent to the mini-batch size)
        # Any value lower than 1 will lead to loading the entire batch
        # mini_batch_size: int = 0,
        batch: int = 0,
        start_run: int = 0,
        start_idx: int = 0,
        initial_batches: list = None,
        use_latent_replay: bool = False,
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
        self.batch = batch
        self.run = start_run
        self.idx = start_idx
        self.randomize_data_order = randomize_data_order
        self.category_based_split = category_based_split
        self.debug_mode = debug_mode
        self.use_latent_replay = use_latent_replay
        self.stored_activations = list()
        self.stored_activations_indexes = list()

        # Native rehearsal variables
        self.rm = list()
        self.rm_size = rehearsal_memory_size
        self.batches_so_far = 0
        self.initial_batches = initial_batches
        self.current_batch = list()

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

        # Get sample from stored activations list, if available
        if self.idx_order[self.idx] in self.stored_activations_indexes:
            # Obtain stored activation
            x, y = self.stored_activations[
                self.stored_activations_indexes.index(self.idx_order[self.idx])
            ]

            # Mark it as activation
            x = (False, x)
        # Get image path and label otherwise
        else:
            # Get file path
            file_path = self.paths[self.idx_order[self.idx]]

            # Determine image path
            img_path = os.path.join(
                self.root,
                "core50_"
                + str(self.original_image_size[0])
                + "x"
                + str(self.original_image_size[1]),
                file_path,
            )

            # Load image
            if not self.debug_mode:
                x = self.get_image_from_path(img_path).astype(np.uint8)

                # Resize image if needed
                # Case 1: larger target image, resize by bordering (equal neutral gray border on either side)
                if self.resize_procedure == ResizeProcedure.BORDER:
                    x = bordering_resize(
                        x,
                        input_image_size=self.input_image_size,
                        original_image_size=self.original_image_size,
                    )

                # Mark it as pattern
                x = (True, x)
            else:
                x = None

            # Determine label
            y = int(file_path.split("/o")[1].split("/C")[0].split("/")[0]) - 1

            # Use category split if required
            if self.category_based_split:
                y = np.array([int(y / 5)], dtype=np.int64)
            else:
                y = np.array([y], dtype=np.int64)

        # Increment current index
        self.idx += 1

        # Don't load images in debug mode
        if self.debug_mode:
            return None, None

        return x, torch.from_numpy(y).to(torch.long)

    def get_image_from_path(self, path):
        # Prepare variable to store the image
        x = np.zeros(
            (
                1,
                self.original_image_size[0],
                self.original_image_size[1],
                self.in_channels,
            ),
            dtype=np.uint8,
        )

        # Load as grayscale
        if self.in_channels == 1:
            x[0, :, :, 0] = np.array(Image.open(path).convert("L"))
        # Load normally
        else:
            x[0] = np.array(Image.open(path))

        return x

    # Function that prepares training batch and randomizes data order
    def do_randomization(self):
        # Get current batch ids
        if isinstance(self.batch, list):
            self.idx_order = list()
            for b in self.batch:
                self.idx_order += list(self.lup[self.scenario][self.run][b])
        else:
            self.idx_order = list(self.lup[self.scenario][self.run][self.batch])

        # Check batch and exemplar set sizes
        if self.debug_mode:
            print()
            print("Batch report")
            print("Batch size:", len(self.idx_order))
            print("Rehearsal memory actual size:", len(self.rm))

        # Store current batch ids
        self.current_batch = self.idx_order.copy()

        # Add exemplar set to id list if needed
        if len(self.rm) > 0 and self.rm[0] not in self.idx_order:
            self.idx_order += self.rm

        # Check final size
        if self.debug_mode:
            print("Total size:", len(self.idx_order))
            print()

        # Randomize their order if required
        if self.randomize_data_order:
            self.idx_order = random.sample(self.idx_order, len(self.idx_order))

        return None

    def update_batch(self, batch):
        # Function to do necessary operations when changing batch
        self.batch = batch
        self.idx = 0
        self.do_randomization()

        return None

    def populate_rm(self):
        # Compute h
        h = self.rm_size // (self.batches_so_far + 1)

        # Treat case of h greater than size of current batch
        if h > len(self.current_batch):
            h = len(self.current_batch)

        # Size of rm to be kept
        n_ext_mem = self.rm_size - h

        # Treat exceptional cases
        assert n_ext_mem >= 0, "Size of rm should never be negative."
        assert h <= len(self.current_batch), "Not enough patterns in current batch."
        assert h <= self.rm_size, "Rehearsal memory size exceeded."

        # Get random h patterns to keep
        if not self.use_latent_replay:
            r_add = random.sample(self.current_batch, h)
        else:
            assert h == len(self.stored_activations), "Latent replay size mismatch."
            r_add = self.stored_activations_indexes

        # Manipulate patterns in rm as required
        if self.batches_so_far in self.initial_batches:
            self.rm = r_add
        else:
            self.rm = random.sample(self.rm, n_ext_mem) + r_add

        # Increment batch counter
        self.batches_so_far += 1

        # In debug mode, check sizes
        if self.debug_mode:
            print()
            print("Rehearsal report")
            print("Required:", h)
            print("To add:", len(r_add))
            print("Maximum allowed rehearsal memory size:", self.rm_size)
            print("Actual rehearsal memory size:", len(self.rm))
            print()

        return None

    def get_classes_in_current_batch(self):
        found_classes = dict()
        for el in self.idx_order:
            current_class = (
                int(self.paths[el].split("/o")[1].split("/C")[0].split("/")[0]) - 1
            )

            if current_class in found_classes:
                found_classes[current_class] += 1
            else:
                found_classes[current_class] = 1

        return found_classes
