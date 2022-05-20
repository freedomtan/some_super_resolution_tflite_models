import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_frozen_pb(saved_model_dir):
  imported = tf.saved_model.load(saved_model_dir)
  model = imported.signatures['serving_default']
  wrapped_model = tf.function(lambda x: imported(x))
  f = wrapped_model.get_concrete_function(
        tf.TensorSpec((1, 1146, 1737, 3), tf.float32))
  frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(f)
  return graph_def

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_model_dir', required=True, help='saved_model directory')
  args = parser.parse_args()

  graph_def = get_frozen_pb(args.saved_model_dir)
  with tf.Graph().as_default() as g:
    tf.graph_util.import_graph_def(graph_def) 
    flops = tf.compat.v1.profiler.profile(g,
            options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    print('total FLOPs: ', flops.total_float_ops)

if __name__ == "__main__":
    main()
