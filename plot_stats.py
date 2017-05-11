import numpy as np
from matplotlib import pyplot as plt
SUMMARY_DIR = './results/qnaf_6'
EXTENTION = '.npy'

results = np.load(SUMMARY_DIR + '/rewards' + EXTENTION)
episodes_xs = np.load(SUMMARY_DIR + '/episodes_xs' + EXTENTION)
episodes_us = np.load(SUMMARY_DIR + '/episodes_us' + EXTENTION)
episodes_rs = np.load(SUMMARY_DIR + '/episodes_rs' + EXTENTION)
episodes_Ps = np.load(SUMMARY_DIR + '/episodes_Ps' + EXTENTION)
episodes_Vs = np.load(SUMMARY_DIR + '/episodes_Vs' + EXTENTION)
#episodes_Qs = np.load(SUMMARY_DIR + '/episodes_Qs' + EXTENTION)
#episodes_Q_s = np.load(SUMMARY_DIR + '/episodes_Q_s' + EXTENTION)
episodes_cs = np.load(SUMMARY_DIR + '/episodes_cs' + EXTENTION)

def plot_episode(episode_num, actual_num):
    ax = plt.subplot(111)
    plt.figure(figsize=(20, 14))

    t = np.arange(len(episodes_xs[episode_num]))
    ax1 = plt.subplot(611)
    us = episodes_us[episode_num].reshape(-1)
    plt.plot(t, us)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('actions')
    ax1.set_ylim(min(us) - 0.1, max(us) + 0.1)
    plt.title('episode # ' + str(episode_num) )

# share x only
    ax2 = plt.subplot(612, sharex=ax1)
    rs = episodes_rs[episode_num].reshape(-1)
    plt.plot(t, rs)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('rewards')
    ax2.set_ylim(min(rs) - 0.1, max(rs) + 0.1)

    ax3 = plt.subplot(613, sharex=ax1)
    Ps = episodes_Ps[episode_num].reshape(-1)
    plt.plot(t, Ps)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('norm of P')
    #ax3.set_ylim(min(Ps) - 0.1, max(Ps) + 0.1)

    ax4 = plt.subplot(614, sharex=ax1)
    Vs = episodes_Vs[episode_num].reshape(-1)
    plt.plot(t, Vs)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylabel('V-estimate')
    ax4.set_ylim(min(Vs) - 0.1, max(Vs) + 0.1)
    plt.xlim(t[0], t[-1])

    '''
    ax5 = plt.subplot(615, sharex=ax1)
    Qs = episodes_Qs[episode_num].reshape(-1)
    plt.plot(t, Qs, label = 'Q-estimate\n(target)')
    Q_s = episodes_Q_s[episode_num].reshape(-1)
    plt.plot(t, Q_s, label='Q-estimate\n(another critic)')
    ax5.set_ylim(min(min(Q_s), min(Qs)) - 0.1, max(max(Q_s), max(Qs)) + 0.1)

    ax5.set_ylabel('Q-est')
    plt.legend()
    plt.xlim(t[0], t[-1])
    plt.setp(ax5.get_xticklabels(), fontsize=6)
    '''

    ax6 = plt.subplot(616)
    cs = np.array(episodes_cs[episode_num]).reshape(-1)
    #for i in range(3):
    #    plt.plot(t, gs[t * 5 + i])
    plt.setp(ax6.get_xticklabels(), fontsize=6)
    ax6.set_ylabel('gradient norm')
    #ax6.set_ylim(min(cs) - 0.1, max(cs) + 0.1)
    plt.xlim(t[0], t[-1])
    plt.plot(t, cs)

    plt.show()
    #plt.savefig('qnaf1' + str(actual_num) + '.png')


def plot_results(episode_nums):
    t = np.arange(len(results))
    plt.plot(t, results)
    plt.xlabel('episode')
    plt.ylabel('R')
    plt.scatter(episode_nums, results[episode_nums])
    plt.vlines(50.5, min(results)-200, max(results) + 500, color='g', alpha=0.5)
    plt.show()
    #plt.savefig('qnaf10.png')
     

episode_nums = [43, 48, 80, 108, 177, 178, 180]
episode_nums = [30, 73, 86, 89, 92]
episode_nums = [0, 1]
#episode_nums = np.arange(1, 10)
plot_results(episode_nums)
for i, episode_num in enumerate(episode_nums):
    plot_episode(episode_num, i + 1)
    #pass

