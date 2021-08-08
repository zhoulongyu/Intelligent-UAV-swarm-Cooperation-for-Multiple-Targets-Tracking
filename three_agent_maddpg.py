import numpy as np
import tensorflow as tf
import pandas as pd
import make_env
import matplotlib.pyplot as plt

from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer
import warnings
warnings.simplefilter("ignore")

def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update

np.seterr(invalid='ignore')

agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

agent4_ddpg = MADDPG('agent4')
agent4_ddpg_target = MADDPG('agent4_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')

agent4_actor_target_init, agent4_actor_target_update = create_init_update('agent4_actor', 'agent4_target_actor')
agent4_critic_target_init, agent4_critic_target_update = create_init_update('agent4_critic', 'agent4_target_critic')


def get_agents_action(o_n, sess, noise_rate=0.):
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(4) * noise_rate/10 # 后面一项产生四个随机数
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(4) * noise_rate/10
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(4) * noise_rate/10
    agent4_action = agent4_ddpg.action(state=[o_n[3]], sess=sess) + np.random.randn(4) * noise_rate/10
    return agent1_action, agent2_action, agent3_action, agent4_action

def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :],total_act_batch[:, 3, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]
    next_other_actor3_o = total_next_obs_batch[:, 3, :]
    # 获取下一个情况下另外两个agent的行动
    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess),
                                   other_actors[1].action(next_other_actor2_o, sess),
                                   other_actors[2].action(next_other_actor3_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    target_loss = agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

    return target_loss

if __name__ == '__main__':
    env = make_env.make_env('simple_env')
    o_n = env.reset()

    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(4)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(4)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(4)]
    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(4)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(4)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(4)]

    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(4)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(4)]

    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(4)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(4)]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])

    summary_writer = tf.summary.FileWriter('./three_ma_summary', graph=tf.get_default_graph())

    agent1_memory = ReplayBuffer(1000)
    agent2_memory = ReplayBuffer(1000)
    agent3_memory = ReplayBuffer(1000)
    agent4_memory = ReplayBuffer(1000)

    # e = 1

    reward_100_list = [[], [], [], []]
    loss_his = {"Agent_1": [], "Agent_2": [], "Agent_3": [], "Agent_4": []}
    for i in range(5000):
        # env.render()
        if i % 100 == 0:
            o_n = env.reset()
            for agent_index in range(4):
                summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                    {reward_1000[agent_index]: np.mean(reward_100_list[agent_index])}),
                                           i // 1000)

        agent1_action, agent2_action, agent3_action, agent4_action = get_agents_action(o_n, sess, noise_rate=0.2)

        #三个agent的行动
        a = np.array([i[0] for i in [agent1_action, agent2_action, agent3_action, agent4_action]])
        # a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]
        # TODO(Weiz): 这里只假设在以为平面进行滑动


        o_n_next, r_n, d_n, i_n = env.step(a, time=i%100)
        print("Average reward : {} ({}/{})".format(r_n, i+1, 10000))
        for agent_index in range(4):
            reward_100_list[agent_index].append(r_n[agent_index])
            # reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

        agent1_memory.add(np.vstack([o_n[0], o_n[1], o_n[2], o_n[3]]),
                          np.vstack([agent1_action[0], agent2_action[0], agent3_action[0], agent4_action[0]]),
                          r_n[0], np.vstack([o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3]]), False)

        agent2_memory.add(np.vstack([o_n[1], o_n[2], o_n[3], o_n[0]]),
                          np.vstack([agent2_action[0], agent3_action[0], agent4_action[0], agent1_action[0]]),
                          r_n[1], np.vstack([o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[0]]), False)

        agent3_memory.add(np.vstack([o_n[2], o_n[3], o_n[0], o_n[1]]),
                          np.vstack([agent3_action[0], agent4_action[0], agent1_action[0], agent2_action[0]]),
                          r_n[2], np.vstack([o_n_next[2], o_n_next[3], o_n_next[0], o_n_next[1]]), False)

        agent4_memory.add(np.vstack([o_n[3], o_n[0], o_n[1], o_n[2]]),
                          np.vstack([agent4_action[0], agent1_action[0], agent2_action[0], agent3_action[0]]),
                          r_n[3], np.vstack([o_n_next[3], o_n_next[0], o_n_next[1], o_n_next[2]]), False)

        if i > 2000 and i % 100 == 0:
            # e *= 0.9999
            # agent1 train
            loss_his['Agent_1'].append(train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                        agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target]))
            loss_his['Agent_2'].append(train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                        agent2_critic_target_update, sess, [agent3_ddpg_target, agent4_ddpg_target, agent1_ddpg_target]))

            loss_his['Agent_3'].append(train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                        agent3_critic_target_update, sess, [agent4_ddpg_target, agent1_ddpg_target, agent2_ddpg_target]))

            loss_his['Agent_4'].append(train_agent(agent4_ddpg, agent4_ddpg_target, agent4_memory, agent4_actor_target_update,
                        agent4_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target]))

        for agent_index in range(4):
            summary_writer.add_summary(
                sess.run(agent_reward_op[agent_index], {agent_reward_v[agent_index]: r_n[agent_index]}), i)
            summary_writer.add_summary(sess.run(agent_a1_op[agent_index], {agent_a1[agent_index]: a[agent_index][1]}),
                                       i)
            summary_writer.add_summary(sess.run(agent_a2_op[agent_index], {agent_a2[agent_index]: a[agent_index][3]}),
                                       i)
            summary_writer.add_summary(
                sess.run(reward_100_op[agent_index], {reward_100[agent_index]: np.mean(reward_100_list[agent_index])}),
                i)

        o_n = o_n_next

        if i % 1000 == 0:
            saver.save(sess, './three_ma_weight/' + str(i) + '.cptk')

    sess.close()
    # critic loss 图
    loss_df = pd.DataFrame(loss_his)
    loss_df.plot()
    plt.title("Loss")

    # reward 图
    reward_his = {}
    for agent_index in range(4):
        reward_his['Agent_{}'.format(agent_index)] = reward_100_list[agent_index]
    reward_df = pd.DataFrame(reward_his)
    reward_df.plot()
    plt.title("Reward")
    plt.show()