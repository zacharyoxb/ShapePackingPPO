""" Reads data from text """

import re

from tensordict import TensorDict
import torch


def _get_presents(file_name) -> torch.Tensor:
    """ Extracts present matrices from data as a PyTorch tensor """
    raw_lines = []
    with open(f"inputs/{file_name}", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    full_text = " ".join(raw_lines)

    # get all shapes matrices from text
    present_pattern = r"\d+:\s*((?:[.#]{3}\s*){3})"
    shape_matches = re.findall(present_pattern, full_text)

    extracted_present_tensors: list[torch.Tensor] = []

    for extracted_matrix in shape_matches:
        split_matrix = extracted_matrix.split()

        binary_matrix: list[list[int]] = [[1 if char == '#' else 0 for char in row]
                                          for row in split_matrix]
        tensor_matrix = torch.tensor(binary_matrix, dtype=torch.float32)
        extracted_present_tensors.append(tensor_matrix)

    return torch.stack(extracted_present_tensors)


def _get_placement_info(file_name="input.txt") -> list[tuple[int, int, torch.Tensor]]:
    """ Gets placement info on area and how many of each present to place """
    raw_lines = []
    with open(f"inputs/{file_name}", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    full_text = " ".join(raw_lines)

    # get all areas we will insert into and indexes of shapes
    region_pattern = r'(\d+)x(\d+):\s*((?:\d+\s*)+)(?=\n|$)'
    region_matches = re.findall(region_pattern, full_text)

    # get height and width of container and amount of presents to place
    args = []
    for width_str, height_str, present_count_str in region_matches:
        width, height = int(width_str), int(height_str)
        present_count_str = present_count_str.split()
        present_count = list(map(int, present_count_str))
        args.append((width, height, torch.tensor(
            present_count, dtype=torch.float32)))
    return args


def get_data(device, file_name="input.txt") -> list[TensorDict]:
    """ Collates all data into a dataset """
    presents = _get_presents(file_name)
    placement_info = _get_placement_info(file_name)

    data = []

    for grid_width, grid_height, present_count in placement_info:
        params = TensorDict({
            "grid": torch.zeros((grid_height, grid_width), dtype=torch.float32),
            "presents": presents.clone(),
            "present_count": present_count.clone(),
        }, batch_size=[], device=device)
        data.append(params)

    return data
