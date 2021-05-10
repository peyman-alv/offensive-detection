import tensorflow as tf
import tensorflow_addons as tfa

from transformers import TFAutoModel

from tensorflow.keras.layers import Input, Reshape, Concatenate, Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout, Conv2D, GlobalMaxPool2D

from config import BERT_NAME, MAX_SEQ_LEN, NUM_CLASSES


def bert(max_len: int, is_trainable: bool) -> tuple:
    """
    Return BERT transformer model, whose inputs are input_ids, token_type_ids, attention_mask.
    Args:
        is_trainable: indicates whether BERT parameters should be updated during training.
    Outputs:
        input_layer: a list of tensorflow Inputs, comprising input_ids, token_type_ids, attention_mask.
        embedding_layer: a matrix with shape of (max_len, 768).
    """
    transformer = TFAutoModel.from_pretrained(BERT_NAME, trainable=is_trainable)

    input_ids      = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.float32, name="input_ids")
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")

    input_layer     = [input_ids, attention_mask, token_type_ids]
    embedding_layer = transformer(input_layer)[0]

    return input_layer, embedding_layer


def nuli(bert_trainable: bool, dropout_rate: float):
    """
    Return Tensorflow format of BertForSequenceClassificatio model. 
    """
    input_layer, embedding_layer = bert(max_len=MAX_SEQ_LEN, is_trainable=bert_trainable)
    dropout1 = Dropout(rate=dropout_rate)(embedding_layer)
    dense1 = Dense(units=NUM_CLASSES, activation='softmax')(dropout1)
    return tf.keras.Model(input_layer, dense1)


def kungfupanda(hidden_size: int, linear_in_features: int, dropout_rate: float, bert_trainable: bool):
    """
    Return Tensorflow format of MTL_Transformer_LSTM model just for subtask_a. 
    """
    input_layer, embedding_layer = bert(max_len=MAX_SEQ_LEN, is_trainable=bert_trainable)

    bilstm_a  = Bidirectional(LSTM(units=hidden_size, return_sequences=True))(embedding_layer)
    concat_a  = Concatenate(axis=1)([bilstm_a[:, 0, :], bilstm_a[:, 1, :]])
    dropout_a = Dropout(dropout_rate)

    dense1 = Dense(units=linear_in_features, activation='relu')(dropout_a)
    dense2 = Dense(units=NUM_CLASSES, activation='softmax')(dense1)

    return tf.keras.Model(input_layer, dense2)


def kusail(bert_trainable: bool, num_filters: int, filter_sizes: list, dropout_rate: float):
    """
    Return Tensorflow format of KUSAIL (BERT-CNN) model. 
    """
    input_layer, embedding_layer = bert(max_len=MAX_SEQ_LEN, is_trainable=bert_trainable)
    reshape1 = Reshape((MAX_SEQ_LEN, 768, 1))(embedding_layer)
    
    conv2d_1 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[0], 768), activation='relu')(reshape1)
    maxpool1 = GlobalMaxPool2D()(conv2d_1)

    conv2d_2 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[1], 768), activation='relu')(reshape1)
    maxpool2 = tf.keras.layers.GlobalMaxPool2D()(conv2d_2)

    conv2d_3 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[2], 768), activation='relu')(reshape1)
    maxpool3 = tf.keras.layers.GlobalMaxPool2D()(conv2d_3)

    conv2d_4 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[3], 768), activation='relu')(reshape1)
    maxpool4 = tf.keras.layers.GlobalMaxPool2D()(conv2d_4)

    conv2d_5 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[4], 768), activation='relu')(reshape1)
    maxpool5 = tf.keras.layers.GlobalMaxPool2D()(conv2d_5)

    concat   = Concatenate(axis=-1)([maxpool1, maxpool2, maxpool3, maxpool4, maxpool5])
    dropout1 = Dropout(rate=dropout_rate)(concat)

    dense2 = Dense(units=NUM_CLASSES, activation='softmax')(dropout1)

    return tf.keras.Model(input_layer, dense2)


def compile_model(model, learning_rate: float, logits: bool, f1_average: str):
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=logits)
    metric = tfa.metrics.F1Score(num_classes=class_num, average=f1_average, name='F1-Macro')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    return model