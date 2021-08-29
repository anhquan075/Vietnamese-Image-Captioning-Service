from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import argparse
# import json
# import codecs

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

import sys
from importlib import reload
reload(sys)

#from ..utils import get_config

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model: Vietnamese Image Captioning""")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Model checkpoint file or directory containing a model checkpoint file.", required=True)
    parser.add_argument("--vocab_file",type=str, help='Text file containing the vocabulary.', required=True)
    parser.add_argument('--input_files', type=str, default="", help="File pattern or comma-separated list of file patterns of image files.", required=True)

    args = parser.parse_args()
    return args


def run(opts):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(), opts.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(opts.vocab_file)

  filename = opts.input_files

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)
    

    with tf.gfile.GFile(filename, "rb") as f:
      image = f.read()
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    
    captions = generator.beam_search(sess, image)
    
    AllCaptions = []
    print(" Captions for image %s:" % (basename))
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      string = "\t{0}) {1} (p={2})".format(i, sentence, math.exp(caption.logprob))
      AllCaptions.append(string)

    return AllCaptions

if __name__ == "__main__":
  opt = get_args()
  print(run(opt))
