__author__ = "Lygerakis Fotios"

import os
import yaml
from tensorflow.python.client import device_lib
import tensorflow as tf
from model import custom_scedule


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def loadCheckpoint_chat(VOCAB_SIZE):
    with open('config.yaml') as f:
        chat_config = yaml.load(f, Loader=yaml.FullLoader)["chat"]
    resume_auto = bool(chat_config["resume_auto"])
    resume_manual = bool(chat_config["resume_manual"])
    manual_ckpt_filename = chat_config["manual_ckpt_filename"]
    language = chat_config["language"].lower()

    checkpoint_path = "checkpoints/"+language+"/"
    if resume_auto:
        checkpoint_filename = "checkpoint_voc_size" + str(VOCAB_SIZE) + \
                              "_" + language + ".ckpt"
        ckpt_dir = checkpoint_path + checkpoint_filename
        return ckpt_dir

    elif resume_manual and manual_ckpt_filename is not None:
        checkpoint_filename = manual_ckpt_filename
        ckpt_dir = checkpoint_path + checkpoint_filename
        return ckpt_dir
    else:
        return None


def loadCheckpoint_train(VOCAB_SIZE, language, model):
    with open('config.yaml') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)["train"]

    resume_auto = bool(train_config["resume_auto"])
    resume_manual = bool(train_config["resume_manual"])
    manual_ckpt_filename = train_config["manual_ckpt_filename"]
    language = train_config["language"].lower()

    checkpoint_path = "checkpoints/"+language+"/"
    if resume_auto:
        checkpoint_filename = "checkpoint_voc_size" + str(VOCAB_SIZE) +\
                              "_" + language + ".ckpt"
        checkpoint_path = checkpoint_path + checkpoint_filename

    elif resume_manual and manual_ckpt_filename is not None:
        checkpoint_filename = manual_ckpt_filename
        checkpoint_path = checkpoint_path + checkpoint_filename

    if os.path.isfile(checkpoint_path + ".index"):
        try:
            model.load_weights(checkpoint_path)
            print("Model loaded from checkpoint " + checkpoint_path + "Loaded")
            return checkpoint_path, model
        except ValueError:
            print("Model could not be loaded. Error: " + str(ValueError))
            print("Creating new model.")
    else:
        print("Checkpoint NOT FOUND.")
        print("Creating new model.")
        return checkpoint_path, model


def getCallbacks(checkpoint_path):
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/scalars/"
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    # Create a callback for schedule learning rate
    lr_callback = tf.keras.callbacks.LearningRateScheduler(custom_scedule)
    # Create a callback for Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    return cp_callback, tensorboard_callback, lr_callback