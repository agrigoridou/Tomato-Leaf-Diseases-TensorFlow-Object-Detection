#   python create_tfrecord.py --csv_input=images/train_labels.csv --output_path=train.record --label_map=images/label_map.pbtxt --img_path=images/train

#   python create_tfrecord.py --csv_input=images/test_labels.csv --output_path=test.record --label_map=images/label_map.pbtxt --img_path=images/test

import os
import io
import pandas as pd
import tensorflow as tf

from object_detection.utils import dataset_util

def create_tf_example(row, image_dir):
    # Read image file
    with tf.io.gfile.GFile(os.path.join(image_dir, row['filename']), 'rb') as f:
        encoded_jpg = f.read()

    # Create TFExample
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(row['height']),
        'image/width': dataset_util.int64_feature(row['width']),
        'image/filename': dataset_util.bytes_feature(row['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(row['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature([row['xmin']]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([row['xmax']]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([row['ymin']]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([row['ymax']]),
        'image/object/class/text': dataset_util.bytes_list_feature([row['class'].encode('utf8')]),
        'image/object/class/label': dataset_util.int64_list_feature([row['class_id']]),
    }))
    return tf_example

def main(csv_file, image_dir, output_file):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Create TFRecord
    writer = tf.io.TFRecordWriter(output_file)
    for index, row in df.iterrows():
        tf_example = create_tf_example(row, image_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print('Successfully created TFRecord:', output_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create TFRecord')
    parser.add_argument('--csv_input', type=str, help='Path to the CSV file')
    parser.add_argument('--output_path', type=str, help='Path to save the TFRecord file')
    parser.add_argument('--label_map', type=str, help='Path to the label map file')
    parser.add_argument('--img_path', type=str, help='Path to the image folder')
    args = parser.parse_args()

    main(args.csv_input, args.img_path, args.output_path)
