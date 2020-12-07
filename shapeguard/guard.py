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

from __future__ import annotations

from copy import copy
from typing import ClassVar, Dict, List, Optional, Union

import attr

from shapeguard import tools


@attr.s(auto_attribs=True)
class ShapeGuard:
    dims: Dict[str, int] = attr.ib(factory=dict)

    _singleton: ClassVar[Optional[ShapeGuard]] = None

    @classmethod
    def get(cls) -> ShapeGuard:
        if cls._singleton is None:
            return cls.reset()
        return cls._singleton

    @classmethod
    def reset(cls) -> ShapeGuard:
        cls._singleton = cls()
        return cls._singleton

    def matches(self, tensor, template: str) -> bool:
        return tools.matches(tensor, template, self.dims)

    def guard(self, tensor, template: str):
        inferred_dims = tools.guard(tensor, template, self.dims)
        self.dims.update(inferred_dims)
        return tensor

    def reshape(self, tensor, template: str):
        return tools.reshape(tensor, template, self.dims)

    def evaluate(self, template: str, **kwargs) -> List[Optional[int]]:
        local_dims = copy(self.dims)
        local_dims.update(kwargs)
        return tools.evaluate(template, local_dims)

    def __getitem__(self, item: str) -> List[Optional[int]]:
        return tools.evaluate(item, self.dims)


def sg(tensor, template: Union[str, List[str]]):

    if isinstance(template, list):
        assert isinstance(
            tensor, list
        ), f"Found list template {template}, but non-list tensor {type(tensor)}"
        assert (
            len(tensor) >= 1
        ), f"Found list template {template}, but empty list tensor"

        if len(template) == 1:
            template = template * len(tensor)

        assert len(template) == len(
            tensor
        ), f"Found {len(template)} templates, but {len(tensor)} tensors"

        for t, m in zip(tensor, template):
            sg(t, m)
    else:
        ShapeGuard.get().guard(tensor, template)
