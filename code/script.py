import gc
import pickle
from typing import Dict

import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as K

EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


def load_embeddings(filepath: str) -> Dict[str, np.array]:
    def _get_vec(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    with open(filepath) as embeddings_file:
        word_embeddings = dict(_get_vec(*line.strip().split(' '))
                               for line in embeddings_file)
        words_to_del = {
            word for word, vec in word_embeddings.items() if len(vec) != 300
        }
        for word in words_to_del:
            del word_embeddings[word]
    return word_embeddings


def build_matrix(word_index: Dict[str, int],
                 word_embeddings_path: str) -> np.array:
    word_embeddings = load_embeddings(word_embeddings_path)

    # get entire embedding matrix
    mat_embedding = np.stack(word_embeddings.values())
    # get shape
    n_words, n_features = len(word_index), mat_embedding.shape[1]
    # init embedding weight matrix
    embedding_mean, embedding_std = mat_embedding.mean(), mat_embedding.std()
    embedding_weights = np.random.normal(embedding_mean, embedding_std, (n_words, n_features))
    # mapping
    for word, idx in word_index.items():
        if idx >= n_words:
            continue
        else:
            word_vec = word_embeddings.get(word, None)
        if word_vec is not None:
            embedding_weights[idx] = word_vec
    # Re-initializing embedding for OOV token
    embedding_weights[1] = embedding_weights[2:].mean(axis=0)
    return embedding_weights


def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:, 0], (-1, 1)), y_pred) * y_true[:, 1]


def build_model(embedding_matrix: np.array, num_aux_targets: int, loss_weight: float) -> Model:
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss=[custom_loss, 'binary_crossentropy'],
                  loss_weights=[loss_weight, 1.0], optimizer='adam')
    return model


def preprocess(text_column: pd.Series) -> pd.Series:
    """
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    """
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    text_column = text_column.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return text_column


if __name__ == '__main__':
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

    x_train = preprocess(train['comment_text'])

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    # Overall
    weights = np.ones((len(x_train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()

    y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
    x_test = preprocess(test['comment_text'])

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    embedding_matrix = np.concatenate(
        [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

    with open('temporary.pickle', mode='wb') as f:
        pickle.dump(x_test, f)  # use temporary file to reduce memory

    del identity_columns, weights, tokenizer, train, test, x_test
    gc.collect()

    checkpoint_predictions = []
    weights = []

    for model_idx in range(NUM_MODELS):
        model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)
        for global_epoch in range(EPOCHS):
            model.fit(
                x_train,
                [y_train, y_aux_train],
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=1,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
                ]
            )
            with open('temporary.pickle', mode='rb') as f:
                x_test = pickle.load(f)  # use temporary file to reduce memory
            checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
            del x_test
            gc.collect()
            weights.append(2 ** global_epoch)
        del model
        gc.collect()

    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

    df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
    df_submit.prediction = predictions
    df_submit.to_csv('submission.csv', index=False)
