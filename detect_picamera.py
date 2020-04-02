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
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

import cv2

from annotation import Annotator

import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers.""" #yepyep
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels  # the labels are here, gets them from a file. kay


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  # a tensor is a vector within a matrix https://www.kdnuggets.com/2018/05/wtf-tensor.html  mkayy...
  # so it just creates an empty tensor to populate later?
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  #okay so get__output_details just says it return a list of output details  (whatever those are...)
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor
# now tensor has stuff in it - kay.

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke() # so invoke is like .append? no... hm

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)  #so  second number is the index, i need some way of visualizing this
  classes = get_output_tensor(interpreter, 1) # are boxes, classes, scores, count, labels? hm   no they are not labels dumba$$
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold: # nope i don't get it...  :(   # kay i watched some videos that helped
      result = {
          'bounding_box': boxes[i],  # this is the box it draws around the object
          'class_id': classes[i],  # this is the name of the object
          'score': scores[i]   # i think this is the confidence percentage
      }
      results.append(result)
  return results


def annotate_objects(annotator, image, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box(image, [xmin, ymin, xmax, ymax])
    annotator.text(image, [xmin, ymin],
                   '%s %.2f' % (labels[obj['class_id']], obj['score']))


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)  # i get an error in the labels and models i forgot how we changed those last time
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required= False,
      type=float,
      default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  cap = cv2.VideoCapture(0)
  annotator = Annotator()

  while True:
    ret, frame = cap.read()
    image = Image.fromarray(frame).convert('RGB').resize((input_width, input_height), Image.ANTIALIAS)

    start_time = time.time()
    results = detect_objects(interpreter, image, args.threshold)
    elapsed_ms = (time.time() - start_time) * 1000
    annotate_objects(annotator, frame, results, labels)
    annotator.text(frame,[5, 30], '%.1fms' % (elapsed_ms))
    cv2.imshow("detectObjects", frame)
    # got above from our last example (check)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()

