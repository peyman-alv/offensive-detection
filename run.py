import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from cli import get_args
from config import SAVED_PATH, TARGET_MAP
from inputs import calcualte_model_io
from model import *

from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt



def print_available_sources() -> None:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.gpu_device_name())


if __name__ == "__main__":
    print_available_sources()

    # Get command line arguments
    args = get_args()
    
    # calculate model inputs
    al, fn, ml, nc = args["apply_likelihood"], args["formula_name"], args["max_len"], args["n_classes"]
    Xtrain, Ytrain, Xtest, Ytest = calcualte_model_io(al, fn, ml, nc)
    
    print("-"*60)
    print(f"Xtrain's shape: {Xtrain.shape} -- Ytrain's shape: {Ytrain.shape} -- \
     Xtest's shape: {Xtest.shape} -- Ytest's shape: {Ytest.shape}")
    print("-"*60)
    
    print(f"One example of input model:\n  input_ids:{Xtrain[0, 0, :]}\n  attention_mask:{Xtrain[0, 1, :]}\n  token_type_ids:{Xtrain[0, 2, :]}")
    print("-"*60)

    # select model
    model_name = args["model_name"]
    bert_trainable = args["bert_trainable"]

    if model_name == "nuli":
        model = nuli(bert_trainable=bert_trainable, dropout_rate=args["nuli_dropout"])
    elif model_name == "kungfupanda":
        hd, lf, kd = args["hidden_size"], args["linear_in_features"], args["kungfupanda_dropout"]
        model = kungfupanda(hidden_size=hd, linear_in_features=lf, dropout_rate=kd, bert_trainable=bert_trainable)
    elif model_name == "kusail":
        nf, fs, ksd =  args["num_filters"], args["filter_sizes"], args["kusail_dropout"]
        model = kusail(bert_trainable, num_filters=nf, filter_sizes=fs, dropout_rate=ksd)
    else:
        raise TypeError(f"The {model_name} does not exit in implemention.")
    
    model = compile_model(model, learning_rate=args["learning_rate"], logits=True, f1_average="macro")
    
    # train the model
    Xtrain = [Xtrain[:, 0, :], Xtrain[:, 1, :], Xtrain[:, 2, :]]
    Xtest  = [Xtest[:, 0, :], Xtest[:, 1, :], Xtest[:, 2, :]]

    filepath = os.path.join(SAVED_PATH, f"{model_name}.hdf5")
    checkpoint= ModelCheckpoint(filepath, monitor='val_F1-Macro', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), \
         batch_size=args["batch_size"], epochs=args["num_epochs"], callbacks=callbacks_list, shuffle=True)

    # represent the results
    model.load_weights(filepath)

    y_true = np.argmax(Ytest, axis=1)
    y_pred = np.argmax(model.predict(Xtest), axis=1)

    classes = list(TARGET_MAP.keys())
    CM = confusion_matrix(y_true, y_pred)

    fig, ax = plot_confusion_matrix(conf_mat=CM, figsize=(8, 8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(len(classes)), classes, fontsize=12)
    plt.yticks(range(len(classes)), classes, fontsize=12)
    plt.show()

    cls_report_print = classification_report(y_true, y_pred, target_names=classes, digits=4)

    print("\n\n")
    print("-"*60)
    print(cls_report_print)
    print("-"*60)