"""
📔 https://blog.onebar.io/building-a-semantic-search-engine-using-open-source-components-e15af5ed7885
"""

import os
import warnings

warnings.simplefilter("ignore", FutureWarning)

import tensorflow as tf
from tensorflow.saved_model import simple_save

import tensorflow_hub as hub
import tf_sentencepiece

tf.logging.set_verbosity(tf.logging.ERROR)

SIMPLE_TENSORFLOW_SERVING_REPO = (
    "https://github.com/tobegit3hub/simple_tensorflow_serving"
)
SENTENCEPIECE_REPO = "https://github.com/google/sentencepiece"
TF_VERSION = "1.13.1"


def clone_repo(url: str):
    os.system(f"rm -rf {url.split('/')[-1]}")
    return os.system(f"git clone {url}")


def download_model():
    export_dir = "./simple_tensorflow_serving/models/use/001"
    with tf.Session(graph=tf.Graph()) as sess:
        module = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1"
        )
        text_input = tf.placeholder(dtype=tf.string, shape=[None])

        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        embeddings = module(text_input)

        simple_save(
            sess,
            export_dir,
            inputs={"text": text_input},
            outputs={"embeddings": embeddings},
            legacy_init_op=tf.tables_initializer(),
        )


def build():
    # Clone repos.
    print("Cloning repos.")
    clone_repo(SIMPLE_TENSORFLOW_SERVING_REPO)
    clone_repo(SENTENCEPIECE_REPO)

    # Create simple_tensorflow_serving -> tf_sentencepiece directory.
    print("Creating tf_sentencepiece directory inside simple_tensorflow_serving.")
    os.system("mkdir simple_tensorflow_serving/tf_sentencepiece")
    os.system("ls simple_tensorflow_serving")

    # Copy the correct version of tf ops.
    print("Copying tf_sentencepiece TensorFlow operation.")
    cmd = (
        "cp sentencepiece/tensorflow/tf_sentencepiece/"
        f"_sentencepiece_processor_ops.so.{TF_VERSION} "
        "simple_tensorflow_serving/tf_sentencepiece/sentencepiece_processor_ops.so"
    )
    os.system(cmd)

    # Pin desidered tensorflow version.
    print("Pinning tensorflow version on requirements.")
    cmd = f"sed -i.back s'/tensorflow.*$/tensorflow=={TF_VERSION}/g' simple_tensorflow_serving/requirements.txt"
    os.system(cmd)

    # Fix Simple Tensorflow Serving's missing template issue.
    # https://github.com/tobegit3hub/simple_tensorflow_serving/pull/70/commits/586a2329324426a4d154f3ffb496eaa523010d69
    cmd = r"sed -i.bak s'/\=\x27templates\x27/\=\"simple_tensorflow_serving\/templates\", static_folder=\"simple_tensorflow_serving\/static\"/g' simple_tensorflow_serving/simple_tensorflow_serving/server.py"
    os.system(cmd)

    # Create a folder to store the USE model.
    print("Creating model directory.")
    os.system("mkdir simple_tensorflow_serving/models/use")

    # Download the model.
    print("Downloading model.")
    download_model()

    # Copy template file to actual dockerfile.
    print("Replacing Dockerfile.")
    os.system(
        "mv simple_tensorflow_serving/Dockerfile simple_tensorflow_serving/Dockerfile.bak"
    )
    os.system("cp Dockerfile.template simple_tensorflow_serving/Dockerfile")

    print("Done.\n")
    print(
        "Remember to \x1b[1;32;40m"
        "cd simple_tensorflow_serving && docker build . --tag stefanondisponibile/simple-tf-use:latest\x1b[0m "
        "to build the container."
    )
    print(
        "And to      \x1b[1;32;40mdocker push stefanondisponibile/simple-tf-use\x1b[0m to push it to DockerHub."
    )
    print(
        "Run         \x1b[1;32;40mdocker run -dp 8501:8501 stefanondisponibile/simple-tf-use:latest\x1b[0m to run the container."
    )
    cmd = "curl -H " \
          "\"Content-Type: application/json\" " \
          "-X POST -d '{\"model_name\": \"default\", \"model_version\": \"001\"," \
          " \"data\": { \"text\": [\"Some text.\"] }}' http://localhost:8501"
    print(
        f"Try it out: \x1b[1;32;40m{cmd}\x1b[0m"
    )


def main():
    build()


if __name__ == "__main__":
    main()
