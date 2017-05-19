import tensorflow as tf
import numpy as np
import gym
from collections import deque
from matplotlib import pyplot as plt
import random
from utils import NormalizedActions
from NAF import NAF


MAX_EP_STEPS = 200
LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.001

RENDER_ENV = False
GYM_MONITOR_EN = True
ENV_NAME = 'Pendulum-v0'
MONITOR_DIR = './results/qnaf_7'
SUMMARY_DIR = './results/tf_qnaf_7'


RANDOM_SEED = 42
BUFFER_SIZE = 800000
MINIBATCH_SIZE = 64

NOISE_MEAN = 0
NOISE_VAR = 1
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
EXPLORATION_TIME = 200
MAX_EPISODES = 500


def main(_):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env = NormalizedActions(gym.make(ENV_NAME))
    env.seed(RANDOM_SEED)
    if GYM_MONITOR_EN:
        if not RENDER_ENV:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
        else:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)
    with tf.Session() as sess:
        monitor_dir = MONITOR_DIR# + str(iteration)

        naf = NAF(sess, env, LEARNING_RATE, TAU, GAMMA,
                     BUFFER_SIZE, RANDOM_SEED, monitor_dir, False, det=False, pg=False, qnaf=False,
                     scope='qn', hn=0, ac=True,
                     sep_V=True, per_st=False)
        naf.run_n_episodes(EXPLORATION_TIME, MAX_EP_STEPS,
                           MINIBATCH_SIZE, num_updates=5, eta=1, num_updates_ac=5)
        naf.run_n_episodes(MAX_EPISODES - EXPLORATION_TIME, MAX_EP_STEPS,
                           MINIBATCH_SIZE, False, num_updates=3, eta=1, num_updates_ac=5)
        naf.plot_rewards(monitor_dir)

if __name__ == '__main__':
    tf.app.run()


