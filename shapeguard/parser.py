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
"""Defines the transformation from a shape template parse tree to ShapeSpec."""

from __future__ import absolute_import, division, print_function

from pathlib import Path
from typing import Callable, Optional

import lark

from shapeguard import dim_specs, shape_spec

GRAMMAR_FILE = Path(__file__).parent / "shape_spec.lark"


class TreeToSpec(lark.Transformer):
    start = shape_spec.ShapeSpec
    wildcard = dim_specs.Wildcard.make
    ellipsis = dim_specs.EllipsisDim.make
    dynamic = dim_specs.Dynamic.make
    name = dim_specs.NamedDim.make
    dynamic_name = dim_specs.DynamicNamedDim.make
    number = dim_specs.Number.make
    add = dim_specs.AddDims.make
    sub = dim_specs.SubDims.make
    mul = dim_specs.MulDims.make
    div = dim_specs.DivDims.make


parser = lark.Lark(
    grammar=GRAMMAR_FILE.read_text(), transformer=TreeToSpec(), parser="lalr"
)
parse: Callable[[str], shape_spec.ShapeSpec] = parser.parse  # type: ignore
