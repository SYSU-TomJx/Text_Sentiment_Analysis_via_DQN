import argparse
import jieba
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from IPython.display import Image
from metrics import metrics
from models import Network, LSTM
from sklearn.model_selection import train_test_split
from sklearn import svm
from tqdm import tqdm


ALPHA = 0.5
BATCH_SIZE = 32
EPOCHS = 3000
EPSILON_TOTAL = 2020
PRE_TRAIN_NUMS = 16
QUE = deque(maxlen=32)
TRAIN_STEPS = 64
TRANSFER_STEP = 16


class Environment(object):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self.sample_index()
        self.action_space = len(set(y)) - 1

    def get_state(self):
        '''
        Sampling randomly if start or no reward.
        '''
        obs, _ = self.step(-1)
        return obs

    def step(self, action):
        '''
        Return the current reward and a sample as the next state.
        '''
        if action == -1:  # At the begining or no reward at current state.
            current_index = self.current_index
            self.current_index = self.sample_index()
            return (self.train_X[current_index], 0)
        r = self.reward(action)
        self.current_index = self.sample_index()
        return self.train_X[self.current_index], r

    def reward(self, action):
        '''
        Return reward 1 if the prediction is correct, otherwise return -1.
        '''
        c = self.train_Y[self.current_index]
        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space)

    def sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)


def predict(model, inputs):
    '''
    Return the position of the max q_outputs as the prediction.
    '''
    q_outputs = model.predict(inputs)
    return np.argmax(q_outputs, axis=1)


def get_q_values(model, state):
    inputs = [state.reshape(1, *state.shape), actions]
    qvalues = model.predict(inputs)
    return qvalues[0]


def epsilon_calc(step, ep_min=0.01, ep_max=1, esp_total=1000):
    return max(ep_min, ep_max - (ep_max - ep_min)*step/esp_total)


def epsilon_greedy(env, state, step, actor, ep_min=0.01, ep_total=1000):
    epsilon = epsilon_calc(step, ep_min, 1, ep_total)
    if np.random.rand()<epsilon:
        return env.sample_actions(),0
    qvalues = get_q_values(actor, state)
    return np.argmax(qvalues), np.max(qvalues)


def seg_word(doc):
    seg_list = jieba.cut(doc, cut_all=False)
    return list(seg_list)


def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)


def get_parser():
    '''
    requirement for execution
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DRL', help='Select a model from DRL, LSTM, SVM', type=str)
    parser.add_argument('--mode', default='train', help='Select a mode from train or eval', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser()
    MODEL = args.model
    MODE = args.mode
    if MODEL not in ['DRL', 'LSTM', 'SVM']:
        raise (Exception('Please select a model from DRL, LSTM or SVM.'))
    if MODE not in ['train', 'eval']:
        raise (Exception('Please select a mode from train or eval.'))
    print(MODEL,' ', MODE)

    with open(r'./weibo_senti_100k/weibo_senti_100k.csv', 'r', encoding='utf-8') as handle:
        df = pd.read_csv(handle)
    print('comments: %d' % df.shape[0])
    print('positive samples: %d' % df[df.label == 1].shape[0])
    print('negative samples: %d' % df[df.label == 0].shape[0])

    comments = df['review'].tolist()
    sentiments = df['label'].tolist()
    num_actions = len(set(sentiments))
    x_train, x_test, y_train, y_test = train_test_split(comments, sentiments, test_size=0.2, random_state=42)
    print('Actions: {}'.format(num_actions))
    print('Train set: {}, Test set: {}'.format(len(y_train), len(y_test)))

    comments_seg = [get_char(x) for x in comments]
    train_seg = [get_char(x) for x in x_train]
    test_seg = [get_char(x) for x in x_test]
    if not os.path.exists(r'./token/chars.vector'):
        word2vec_model = Word2Vec(comments_seg, size=100, window=10, min_count=1, workers=4, iter=15)
        word2vec_model.wv.save_word2vec_format(r'./token/chars.vector', binary=True)

    # Take one-hot encoding for sentiment label.
    actions = np.ones((1, num_actions))

    tr_one_hot = tf.keras.utils.to_categorical(y_train, num_actions)
    ts_one_hot = tf.keras.utils.to_categorical(y_test, num_actions)

    # Word embedding
    maxlen = 100
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None)
    tokenizer.fit_on_texts(comments_seg)
    with open(r'./token/tokenizer_chars.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    word_index = tokenizer.word_index
    w2_model = KeyedVectors.load_word2vec_format(r'./token/chars.vector',
                                                 binary=True,
                                                 encoding='utf8',
                                                 unicode_errors='ignore')
    embeddings_index = {}
    embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
    word2idx = {"_PAD": 0}
    vocab_list = [(k, w2_model.wv[k]) for k, v in w2_model.wv.vocab.items()]

    for word, i in word_index.items():
        if word in w2_model:
            embedding_vector = w2_model[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    list_tokenized_train = tokenizer.texts_to_sequences(train_seg)

    # Train data, max length is set to 100
    input_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train,
                                                                maxlen=maxlen)

    list_tokenized_validation = tokenizer.texts_to_sequences(test_seg)
    # Test data, max length is set to 100
    input_validation = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_validation,
                                                                     maxlen=maxlen)

    # Network input layers
    inputs = tf.keras.Input(shape=(maxlen,),
                            name='embds_input')
    act_input = tf.keras.Input((num_actions,),
                               name='actions_input')
    if MODEL == 'DRL':
        # Create an environment
        ENV = Environment(input_train, y_train)

        # Actor-Critic

        # Train
        critic_model = Network(num_actions, embeddings_matrix, maxlen=maxlen)
        _ = critic_model(inputs=[inputs, act_input])
        critic_model.compile(loss='mse', optimizer='adam')

        # Decision
        actor_q_model = tf.keras.Model(inputs=critic_model.input, outputs=critic_model.get_layer('q_outputs').output)

        if MODE == 'train':

            def train(samples):

                if len(samples) < BATCH_SIZE:
                    return
                samples = np.array(samples)
                states, actions, old_q, rewards, next_states = zip(*samples)
                states, actions, old_q, rewards = np.array(states), np.array(actions).reshape(-1, 1),\
                                                  np.array(old_q).reshape(-1, 1), np.array(rewards).reshape(-1, 1)
                actions_one_hot = tf.keras.utils.to_categorical(actions, num_actions)

                q_estimate = (1-ALPHA) * old_q + ALPHA * rewards.reshape(-1,1)
                history = critic_model.fit([states,actions_one_hot],
                                           q_estimate,
                                           batch_size=BATCH_SIZE,
                                           epochs=1,
                                           verbose=0,
                                           shuffle=True)
                return np.mean(history.history['loss'])

            def pre_enque(pre_go=30):

                state = ENV.get_state()
                for i in range(pre_go):
                    rdn_action = ENV.sample_actions()
                    next_state, reward = ENV.step(rdn_action)
                    QUE.append([state, rdn_action, 0, reward, next_state])
                    state = next_state

            # Training

            process_bar = tqdm(range(1, EPOCHS + 1))
            total_rewards = 0
            reward_rec = []
            QUE.clear()
            state = ENV.get_state()
            pre_enque(PRE_TRAIN_NUMS)
            for epoch in process_bar:
                total_rewards = 0
                epo_start = time.time()
                for step in range(TRAIN_STEPS):
                    # Epsilon greedy for each state
                    action, q = epsilon_greedy(ENV, state, epoch, actor_q_model, ep_min=0.01,
                                               ep_total=EPSILON_TOTAL)
                    eps = epsilon_calc(epoch, esp_total=EPSILON_TOTAL)
                    # Obtain the current reward and next state.
                    next_state, reward = ENV.step(action)
                    # Add the above contain into a queue
                    QUE.append([state, action, q, reward, next_state])
                    # Train
                    loss = train(QUE)
                    total_rewards += reward
                    state = next_state
                    if step % TRANSFER_STEP == 0:
                        actor_q_model = tf.keras.Model(inputs=critic_model.input,
                                                       outputs=critic_model.get_layer('q_outputs').output)
                reward_rec.append(total_rewards)
                process_bar.set_description('Reward:{} Loss:{:.4f} Epsilon:{:.3f} Time:{} '.format(total_rewards, loss, eps,
                                                                                                   int(time.time() - epo_start)))
            # Plot
            cruve = np.mean([reward_rec[i:i + 10] for i in range(0, len(reward_rec), 10)], axis=1)
            plt.plot(range(len(cruve)), cruve, c='b')
            plt.xlabel(r'Epochs $ \times 10$')
            plt.ylabel('Rewards')

            # Save model
            critic_model.save_weights(r'./saved_model/DRL/crtic_3000.HDF5')
            print('model saved.')

        elif MODE == 'eval':
            # Load model
            critic_model = Network(num_actions, embeddings_matrix)
            _ = critic_model(inputs=[inputs, act_input])
            critic_model.load_weights(r'./saved_model/DRL/crtic_3000.HDF5')
            print('model loaded.')
        actor_q_model = tf.keras.Model(inputs=critic_model.input, outputs=critic_model.get_layer('q_outputs').output)

        ones = np.ones(shape=(len(test_seg), num_actions))
        pred = predict(actor_q_model, (input_validation, ones))
        print('On validate data: ')
        metrics(y_test, pred)

        ones_tr = np.ones(shape=(len(train_seg), num_actions))
        pred_tr = predict(actor_q_model, (input_train, ones_tr))
        print('On train data: ')
        metrics(y_train, pred_tr)

    elif MODEL == 'LSTM':

        class_nums = 2
        lstm = LSTM(class_nums, embeddings_matrix, maxlen=maxlen)
        _ = lstm(inputs=inputs)
        adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        rmsprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        lstm.compile(loss='categorical_crossentropy', optimizer=adam)
        if MODE == 'train':
            lstm.fit(input_train, tr_one_hot, batch_size=32, epochs=20, verbose=1, shuffle=True)

            # Save model
            lstm.save_weights(r'./saved_model/LSTM/lstm_20.HDF5')
            print('model saved.')
        else:
            # Load model
            lstm.load_weights(r'./saved_model/LSTM/lstm_20.HDF5')
            print('model loaded.')

        pred = lstm.predict(input_validation)
        pred = np.argmax(pred, axis=1)
        print('On validate data: ')
        metrics(y_test, pred)

        pred_tr = lstm.predict(input_train)
        pred_tr = np.argmax(pred_tr, axis=1)
        print('On train data: ')
        metrics(y_train, pred_tr)

    # SVM
    else:
        tr_data = [np.mean(embeddings_matrix[microblog], axis=0) for microblog in input_train]
        ts_data = [np.mean(embeddings_matrix[microblog], axis=0) for microblog in input_validation]

        cls = svm.SVC()
        cls.fit(tr_data, y_train)

        svm_pred = cls.predict(ts_data)
        svm_pred_tr = cls.predict(tr_data)

        svm_pred = svm_pred.tolist()
        svm_pred_tr = svm_pred_tr.tolist()

        print('On validate data: ')
        metrics(y_test, svm_pred)

        print('On train data: ')
        metrics(y_train, svm_pred_tr)
