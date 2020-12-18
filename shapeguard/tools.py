# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the main ShapeGuard class."""

from copy import copy
from typing import Any, Dict, List, Optional

from shapeguard import exception

from .parser import parse
from .shims import get_shim

Tensor = Any


def matches(tensor: Tensor, template: str, dims: Dict[str, int]) -> bool:
    shape = get_shape(tensor)
    spec = parse(template)
    return spec.matches(shape, dims)


def reshape(tensor: Tensor, template: str, dims: Dict[str, int]) -> Tensor:
    spec = parse(template)
    new_shape = spec.evaluate(dims)
    shim = get_shim(tensor)
    return shim.reshape(new_shape)


def evaluate(template: str, dims: Dict[str, int]) -> List[Optional[int]]:
    dim_spec = parse(template)
    return dim_spec.evaluate(dims)


def guard(tensor: Tensor, template: str, dims: Dict[str, int]):
    shape = get_shape(tensor)
    spec = parse(template)
    # compare rank
    if not spec.rank_matches(shape):
        raise exception.ShapeError(
            "Tensor has the wrong rank ({} != {}).\n"
            "Expected shape: {} (from template {})\n"
            "  Actual shape: {}".format(
                len(shape), len(spec), spec.partial_evaluate(dims), template, shape
            )
        )
    # infer dimensions
    inferred_dims = spec.infer(shape, dims)
    known_dims = copy(dims)
    known_dims.update(inferred_dims)
    # check if dimensions match
    if not spec.matches(shape, known_dims):
        raise exception.ShapeError(
            "Shape Mismatch\n"
            "Expected shape: {} (from template {})\n"
            "  Actual shape: {}".format(spec.partial_evaluate(dims), template, shape)
        )

    # return the inferred dims unless they start with '_'
    return {k: v for k, v in inferred_dims.items() if not k.startswith("_")}


def get_shape(tensor: Tensor) -> List[int]:
    shim = get_shim(tensor)
    return shim.get_shape()
