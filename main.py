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
MONITOR_DIR = './results/qnaf_12'
SUMMARY_DIR = './results/tf_qnaf_12'
#2 pure det qnaf
#1 pure norm qnaf 
#3 pure norm dep qnaf
#4 pure norm dep L qnaf
#5 qnaf + pg nodep norm #TODO ?
#6 qnaf P_inv dep norm
#7 pg norm nodep
#8 pg norm nodep tanh, all previous pg stuff was bugged
#9 pg norm nodep tanh
#10 #FIXME everything before with pg was kinda wrong, pg norm nodep
#11 pg norm nodep lr = 1e-4
#12 reinforce
#13 actor-critic


RANDOM_SEED = 1234
BUFFER_SIZE = 800000
MINIBATCH_SIZE = 64

NOISE_MEAN = 0
NOISE_VAR = 1
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
EXPLORATION_TIME = 150
MAX_EPISODES = 500

PURE_QNAF = True

def main(_):
    with tf.Session() as sess:
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env = NormalizedActions(gym.make(ENV_NAME))
        env.seed(RANDOM_SEED)
        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)

        naf = NAF(sess, env, LEARNING_RATE, TAU, GAMMA,
                     BUFFER_SIZE, RANDOM_SEED, MONITOR_DIR, False, det=False, pg=False, qnaf=False)
        naf.run_n_episodes(EXPLORATION_TIME, MAX_EP_STEPS,
                           MINIBATCH_SIZE, num_updates=5)
        naf.run_n_episodes(MAX_EPISODES - EXPLORATION_TIME, MAX_EP_STEPS,
                           MINIBATCH_SIZE, False, num_updates=3)
        naf.plot_rewards(MONITOR_DIR)

if __name__ == '__main__':
    tf.app.run()

