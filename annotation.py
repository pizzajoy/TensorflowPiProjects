# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An annotation library that draws overlays on the Pi camera preview.

Annotations include bounding boxes and text overlays.
Annotations support partial opacity, however only with respect to the content in
the preview. A transparent fill value will cover up previously drawn overlay
under it, but not the camera content under it. A color of None can be given,
which will then not cover up overlay content drawn under the region.
Note: Overlays do not persist through to the storage layer so images saved from
the camera, will not contain overlays.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2


def _round_up(value, n):
  """Rounds up the given value to the next number divisible by n.

  Args:
    value: int to be rounded up.
    n: the number that should be divisible into value.

  Returns:
    the result of value rounded up to the next multiple of n.
  """
  return n * ((value + (n - 1)) // n)


def _round_buffer_dims(dims):
  """Appropriately rounds the given dimensions for image overlaying.

  As per the PiCamera.add_overlay documentation, the source data must have a
  width rounded up to the nearest multiple of 32, and the height rounded up to
  the nearest multiple of 16. This does that for the given image dimensions.

  Args:
    dims: image dimensions.

  Returns:
    the rounded-up dimensions in a tuple.
  """
  width, height = dims
  return _round_up(width, 32), _round_up(height, 16)


class Annotator:
  """Utility for managing annotations on the camera preview."""

  def __init__(self, default_color=None):
    """Initializes Annotator parameters.

    Args:
      camera: picamera.PiCamera camera object to overlay on top of.
      default_color: PIL.ImageColor (with alpha) default for the drawn content.
    """
    self._default_color = default_color or (0, 0, 0xFF)

  def bounding_box(self, image, rect, outline=None, fill=None):
    """Draws a bounding box around the specified rectangle.

    Args:
      rect: (x1, y1, x2, y2) rectangle to be drawn, where (x1, y1) and (x2, y2)
        are opposite corners of the desired rectangle.
      outline: PIL.ImageColor with which to draw the outline (defaults to the
        Annotator default_color).
      fill: PIL.ImageColor with which to fill the rectangle (defaults to None,
        which will *not* cover up drawings under the region).
    """
    p1 = tuple(rect[:2])
    p2= tuple(rect[2:])
    outline = outline or self._default_color
    cv2.rectangle(image, p1, p2 , outline, 2)

  def text(self, image, location, text, color=None):
    """Draws the given text at the given location.

    Args:
      location: (x, y) point at which to draw the text (upper left corner).
      text: string to be drawn.
      color: PIL.ImageColor to draw the string in (defaults to the Annotator
        default_color).
    """
    color = color or self._default_color
    cv2.putText(image, text, tuple(location), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
