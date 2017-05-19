import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from Network import Network
from Noise import OUNoise
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
from collections import defaultdict


class NAF:
    def __init__(self, sess, env, learning_rate, tau, gamma,
                 buffer_size, random_seed, summary_dir, sigma_P_dep, det, pg, qnaf, scope, hn=0, ac=True, sep_V=True, per_st=False):
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.random_seed = random_seed
        self.sess = sess
        self.summary_dir = summary_dir
        self.det = det
        
        trainables_pre_naf = len(tf.trainable_variables())
        self.model = Network(sess, self.s_dim, self.a_dim, learning_rate,
                             trainables_pre_naf, scope + 'model', sigma_P_dep, det, hn)

        trainables_model = len(tf.trainable_variables())
        self.target_model = Network(sess, self.s_dim, self.a_dim, learning_rate,
                                    trainables_model, scope + 'tmodel', sigma_P_dep, det, hn)
        
        self.target_model.make_soft_update_from(self.model, self.tau)

        trainables_target = len(tf.trainable_variables())
        if sep_V:
            trainables_pre_naf = len(tf.trainable_variables())
            self.critic = Network(sess, self.s_dim, self.a_dim, learning_rate,
                                 trainables_pre_naf, scope + 'cmodel', False, True, 0)
            

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        
        self.buffer = ReplayBuffer(buffer_size, random_seed)
        self.saver = tf.train.Saver()

        #create lists to contain total rewards and steps per episode
        self.rewards = []
        self.episodes_xs = []
        self.episodes_us = []
        self.episodes_rs = []
        self.episodes_Ps = []
        self.episodes_Vs = []
        self.episodes_Qs = []
        self.episodes_Q_s = []
        self.episodes_gs = []
        self.episodes_ss = []
        self.episodes_cs = []
        self.env.reset()
        self.PG = pg
        self.qNAF = qnaf
        self.dep = sigma_P_dep
        self.reinforce = False
        self.ac = ac
        self.calc_c = qnaf and self.dep and (not self.det) and (hn == 0)
        self.separate_V = sep_V
        self.per_st = per_st
        print("end init")


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def run_n_episodes(self, num_episodes, max_ep_length,
                       minibatch_size, explore=True, num_updates = 5, summary_checkpoint=1, eta=0.01, num_updates_ac=1): #num_updates from article
        for i in range(num_episodes):
            noise = OUNoise(self.a_dim) 
            x = self.env.reset()
            x = x.reshape(1, -1)
            u = np.zeros(self.s_dim)
            t = False
            episode_reward = 0
            episode_xs = []
            episode_us = []
            episode_rs = []
            episode_Ps = []
            episode_Vs = []
            episode_Qs = []
            episode_Q_s = []
            episode_gs = []
            episode_ss = []
            episode_cs = []

            #for REINFORCE
            r_xs = []
            r_us = []
            r_rs = []
            #for j in range(max_ep_length):
            while(True):
                if self.det:
                    u = self.sess.run(self.model.mu_det,
                                      feed_dict={self.model.inputs_x: x})
                    if self.qNAF:
                        P, V = self.sess.run((self.model.P, self.model.V),
                                             feed_dict={self.model.inputs_x: x})
                        episode_Ps.append(np.linalg.norm(P))
                        episode_Vs.append(V)

                else:
                    u = self.sess.run(self.model.mu_norm,
                                      feed_dict={self.model.inputs_x: x})
                    if self.qNAF:
                        sigma, P, V = self.sess.run((self.model.sigma, self.model.P, self.model.V),
                                             feed_dict={self.model.inputs_x: x})
                        episode_Ps.append(np.linalg.norm(P))
                        episode_Vs.append(V)
                        episode_ss.append(sigma)
                    if self.calc_c:
                        c = self.sess.run(self.model.C,
                                          feed_dict={self.model.inputs_x: x})
                        episode_cs.append(c)
                if explore:
                    u += noise.noise()
                    u = np.clip(u, -1.0, 1.0)

                if self.qNAF:
                    episode_Qs.append(self.target_model.predict_Q(x, u))             
                if self.PG:
                    episode_Q_s.append(self.critic.predict(x, u.reshape(1, -1)))

                u = u.reshape(1, -1)
                x1, r, t, info = self.env.step(u.reshape(-1))
                r_xs.append(x)
                r_us.append(u)
                r_rs.append(r)
                episode_reward += r
                self.buffer.add(x.reshape(1, -1), u, r, t, x1.reshape(1, -1))
                episode_xs.append(x)
                episode_us.append(u)
                episode_rs.append(r)
                #Actor-Critic
                x = x1.reshape(1, -1)
                
                if self.PG or self.qNAF:
                    for k in range(num_updates):
                        x_batch, u_batch, r_batch, t_batch, x1_batch = \
                            self.buffer.sample_batch(minibatch_size)
                        x_batch, u_batch, r_batch, t_batch, x1_batch = \
                            x_batch.reshape(-1, self.s_dim), u_batch.reshape(-1, self.a_dim), r_batch.reshape(-1, 1),\
                             t_batch.reshape(-1), x1_batch.reshape(-1, self.s_dim)
                        
                        if self.qNAF:
                            y_batch = self.gamma * self.target_model.predict_V(x1_batch) + r_batch
                            self.model.update_Q(x_batch, u_batch, y_batch)
                            
                        self.target_model.soft_update_from(self.model)
                if t:
                    break
            if self.ac:
                r_rs_ = np.array(r_rs).reshape(-1, 1)
                r_xs_ = np.array(r_xs).reshape(-1, self.s_dim)
                r_us_ = np.array(r_us).reshape(-1, self.a_dim)
                for _ in range(num_updates_ac):
                    Q_target = r_rs_[:-1] + self.gamma * self.critic.predict_V_sep(r_xs_[1:])
                    Q_target = np.vstack((Q_target, (r_rs_[-1])))
                    self.model.update_mu(r_xs_, r_us_, Q_target - self.critic.predict_V_sep(r_xs_))
                    self.model.update_V_sep(r_xs_, Q_target)
                    self.target_model.soft_update_from(self.model)



            self.episodes_rs.append(episode_rs)
            self.episodes_us.append(episode_us)
            self.episodes_xs.append(episode_xs)
            self.episodes_Ps.append(episode_Ps)
            self.episodes_Vs.append(episode_Vs)
            self.episodes_Qs.append(episode_Qs)
            self.episodes_ss.append(episode_ss)
            self.episodes_cs.append(episode_cs)
            if self.PG:
                self.episodes_Q_s.append(episode_Q_s)
                self.episodes_gs.append(episode_gs)
            if summary_checkpoint > 0 and i % summary_checkpoint == 0:
                print ('| Reward: %.2i' % int(episode_reward), " | Episode", i)
                self.plot_rewards(self.summary_dir)
            self.rewards.append(episode_reward)

    def plot_rewards(self, summary_dir):
        rewards = np.array(self.rewards).reshape(-1)
        np.save(summary_dir + '/rewards', rewards)
        np.save(summary_dir + '/episodes_xs', np.array(self.episodes_xs))
        np.save(summary_dir + '/episodes_us', np.array(self.episodes_us))
        np.save(summary_dir + '/episodes_rs', np.array(self.episodes_rs))
        np.save(summary_dir + '/episodes_Ps', np.array(self.episodes_Ps))
        np.save(summary_dir + '/episodes_Vs', np.array(self.episodes_Vs))
        np.save(summary_dir + '/episodes_Qs', np.array(self.episodes_Qs))
        np.save(summary_dir + '/episodes_Q_s', np.array(self.episodes_Q_s))
        np.save(summary_dir + '/episodes_gs', np.array(self.episodes_gs))
        np.save(summary_dir + '/episodes_ss', np.array(self.episodes_ss))
        np.save(summary_dir + '/episodes_cs', np.array(self.episodes_cs))

        #plt.plot(np.arange(len(rewards)), rewards)
        #plt.show()
