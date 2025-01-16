#DRQN+RNN(GRU)-PPO MA
from csv import reader

import numpy as np
import torch
# 检查CUDA是否可用
def check_cuda():
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        return True
    else:
        print("No CUDA devices available, using CPU")
        return False
from mec_env_test import Offload
import os
import argparse
import random
from MA_DQN import DRQN, EpisodeBuffer
import gc
import time



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
# setup_seed(23)


def train(RL, RL_wait, args):

    TRANSMIT = args.TRANSMIT
    device = torch.device(args.cuda)
    # agent_memory = list()
    # for iot in range(env.n_iot):
    #     agent_memory.append(EpisodeMemory())    



    reward_average_list = list()
    aoi_average_list = list()
    gamma_list = list()
    # drop_ratio_list = list()   

    duration_list = np.zeros([env.n_iot, env.n_actions])
    duration_average_list = np.zeros([env.n_iot, env.n_actions])
    duration_count_list = np.zeros([env.n_iot, env.n_actions], dtype=int)
    wait_average_list = list()
    ori_average_list = list()
    actor_loss_average = list()
    critic_loss_average = list()
    d3qn_loss_average = list()
    transmit_average = list()
    folder = args.folder
    sub_folder = args.subfolder
    filename1 = 'aoi_average_list.txt'
    filename2 = 'reward_average_list.txt'
    filename3 = 'gamma_list.txt'
    # filename4 = 'drop_ratio_list.txt'
    filename5 = 'wait_action_list.txt'
    filename6 = 'ori_action_list.txt'
    filename7 = 'actor_loss.txt'
    filename8 = 'critic_loss.txt'

    if args.TRANSMIT:
        filename10 = 'transmit_time.txt'

    folderpath =  folder + '/' + sub_folder + '/'
    
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)



    filepath1 = folderpath + filename1
    filepath2 = folderpath + filename2
    filepath3 = folderpath + filename3
    # filepath4 = folderpath + filename4
    filepath5 = folderpath + filename5
    filepath6 = folderpath + filename6
    filepath7 = folderpath + filename7
    filepath8 = folderpath + filename8
    if TRANSMIT:    
        filepath10 = folderpath + filename10
    args_file = folderpath + 'args.txt'
    # np.savetxt(args_file, args)
    args_dict = args.__dict__
    with open(args_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    episode_time_list = list()
    for episode in range(args.num_episode):
        start_time = time.time()
        env.reset(RL)

        #Reset replay buffer
        agent_record = list()
        for iot in range(env.n_iot):
            agent_record.append(EpisodeBuffer())

        step_wait = np.zeros(env.n_iot, dtype=int)
        reward_all = np.zeros(env.n_iot)
        step = np.zeros(env.n_iot, dtype=int)
        step0 = 0
        # drop_all = 0
        
        gc.collect()
        torch.cuda.empty_cache()

        current_wait = np.zeros(env.n_iot)
        count_all = 0
        current_wait_store = list()

        #initialize hidden state for DRQN, GRU in PPO
        # rnn_state = torch.zeros([1, args.rnn_input_size]).to(device)
        # h = RL[iot].q.init_hidden_state(batch_size=args.batch_size, training=False).to(device)  
        # h, c = RL[iot].q.init_hidden_state(batch_size=args.batch_size, training=False)   
        hidden_state = torch.zeros([1, args.rnn_input_size])
        hidden_state_c = torch.zeros([1, args.rnn_input_size])
        ppo_rnn_state = torch.zeros([1, args.rnn_hidden_dim]).to(device)
        ppo_hidden_state = torch.zeros([1, args.rnn_hidden_dim]).to(device)

        for i in range(env.n_iot):
            current_wait_store.append(np.zeros(env.n_iot))
        
            
        last_wait_store = list()
        for i in range(env.n_iot):
            last_wait_store.append(np.zeros(env.n_iot))
        wait_action_list = list()
        ori_action_list = list()
        actor_loss_list = list()
        critic_loss_list = list()
        d3qn_loss_list = list()
        transmit_list = list()

        state_list = list()
        a_logprob_list = list()
        a_wait_logprob_list = list()
        state_val_list = list()
        ar_list = list()
        for i in range(args.num_iot):
            state_list.append(list())
            a_logprob_list.append(list())
            a_wait_logprob_list.append(list())
            state_val_list.append(list())
            ar_list.append(list())
        
        #d3qn epsilon
        epislon = np.zeros(env.n_iot)
        if episode <= args.OFFLOAD_EXPLORE:
            for iot in range(env.n_iot):
                if args.OFFLOAD_EXPLORE != 0:
                    epislon[iot] = 1 - episode / args.OFFLOAD_EXPLORE
                else:
                    epislon[iot] = 0
        for iot in range(env.n_iot):
            RL[iot].epsilon = 0


        f = 0
        while(True):   
            if env.current_time > args.num_time:
                break
            else:
                terminal = False
                done = False

            # RL take action and get next observation and reward
            for i in range(env.n_iot):
                current_wait[i] -=  env.task_iot[0][3]
                if current_wait[i] < 0:
                    current_wait[i] = 0          

            while True:
                render_result = env.render()
                if render_result is not None:
                    break
            current_iot, current_state, wait_state = render_result


            if env.wait_mode[current_iot] == 0:
                # saving reward and is_terminals 
                
                if WAIT and step_wait[current_iot] > 0:
                    RL_wait[current_iot].buffer_new.states.append(state_list[current_iot])
                    RL_wait[current_iot].buffer_new.actions.append(ar_list[current_iot])
                    RL_wait[current_iot].buffer_new.logprobs.append(a_wait_logprob_list[current_iot])
                    RL_wait[current_iot].buffer_new.state_values.append(state_val_list[current_iot])
                    # RL_wait[current_iot].buffer_new.hidden_states_c.append(ppo_hidden_state)
                    RL_wait[current_iot].buffer_new.rewards.append(env.wait_reward)
                    RL_wait[current_iot].buffer_new.is_terminals.append(done)      
                    RL_wait[current_iot].buffer_new.count += 1 

                agent_record[current_iot].put([env.observation, env.action, env.reward, env.observation_next,terminal])                                                        
                reward_all[env.current_iot] += env.wait_reward

  
                    
                if WAIT:
                    if args.model_training == 1 and RL_wait[current_iot].buffer_new.count==args.wait_batch_size:
                        if RL_wait[current_iot].buffer.count == 0 :
                            ppo_rnn_old = torch.zeros([1, args.rnn_hidden_dim]).to(device)
                            ppo_hidden_old = torch.zeros([1, args.rnn_hidden_dim]).to(device)
                            RL_wait[current_iot].buffer = RL_wait[current_iot].buffer_new
                            RL_wait[current_iot].update(ppo_rnn_old, ppo_hidden_old)                          
                            RL_wait[current_iot].buffer_new.clear()
                            ppo_rnn = ppo_rnn_state
                            ppo_hidden = ppo_hidden_state
                        else:
                            ppo_rnn_old = ppo_rnn
                            ppo_hidden_old = ppo_hidden
                            ppo_rnn = ppo_rnn_state
                            ppo_hidden = ppo_hidden_state
                            RL_wait[current_iot].update(ppo_rnn_old, ppo_hidden_old)
                            RL_wait[current_iot].buffer = RL_wait[current_iot].buffer_new
                            RL_wait[current_iot].buffer_new.clear()

                    wait_action, a_wait_logprob_list[current_iot],state_list[current_iot], state_val_list[current_iot], ar_list[current_iot], ppo_rnn_state, ppo_hidden_state =\
                        RL_wait[current_iot].select_action(env.wait_state, ppo_rnn_state, ppo_hidden_state)
                    wait_action = (wait_action / 2 + 0.5) * args.action_range
                    step_wait[current_iot] += 1  
                         
                                          
                else:
                #D3QN only
                    wait_action = args.wait_time

            
                wait_action_list.append(wait_action)
                # ori_action_list.append(ori_action)

                current_wait[current_iot] = wait_action
                env.execute_wait(wait_action)
                                                                                                              


 
            else:
                # Offload
                
                # action = env.auto_action( env.queue_length_edge)
                action = RL[current_iot].q.sample_action(current_state, epislon[current_iot])

                #action = np.random.randint(env.n_edge+1)        
                if action == 0:
                    process_duration, expected_time = env.iot_process(env.n_size, env.comp_cap_iot, env.comp_density)
                    

                else:
                    current_edge = action - 1
                    process_duration, expected_time = env.edge_process(env.n_size, env.comp_cap_edge[current_edge], env.comp_density)
                
                if process_duration < args.drop_coefficient * duration_average_list[current_iot][action]:
                    env.Start_Time[current_iot] = env.current_time
                    env.execute_offload(action, process_duration)
                    # drop_all += 1
                    count_all += 1
                else: #task drop
                    original_wait = env.wait_time[current_iot]   
                    env.execute_wait(process_duration)
                    env.wait_time[current_iot] += original_wait
                    count_all += 1
                duration_list[current_iot][action] += process_duration
                duration_count_list[current_iot][action] += 1
                duration_average_list[current_iot][action] = duration_list[current_iot][action] / duration_count_list[current_iot][action]
                step[current_iot] += 1

            memorylen = len(agent_record[env.current_iot])
            if args.model_training == 1 and memorylen > 0 and memorylen % args.batch_size == 0:
                RL[env.current_iot].train(agent_record[env.current_iot], device)  
                agent_record[env.current_iot].clear()          
                    

                    
        # episode_time = time.time() - time1
        # print("Episode Time: "+str(episode_time))
        # for i in range(env.n_iot):
        #     agent_memory[i].put(agent_record[i])
        aoi_average = 0
        
        reward_average = 0

        for iot in range(env.n_iot):
            reward_average += reward_all[iot] / step[iot]

        reward_average /= env.n_iot

        aoi_average = np.mean(env.aoi_average)

        if len(actor_loss_list) > 0:
            actor_average = sum(actor_loss_list)/len(actor_loss_list)
            actor_loss_average.append(actor_average)

            critic_average = sum(critic_loss_list)/len(critic_loss_list)
            critic_loss_average.append(critic_average)

            d3qn_average = sum(d3qn_loss_list) / len(d3qn_loss_list)
            d3qn_loss_average.append(d3qn_average)

        reward_average_list.append(reward_average)
        aoi_average_list.append(aoi_average)
        gamma_list.append(env.gamma)



        if len(wait_action_list) > 0:
            wait_average = sum(wait_action_list)/len(wait_action_list)
            wait_average_list.append(wait_average)
        if len(ori_action_list) > 0:
            ori_average = sum(ori_action_list)/len(ori_action_list)
            ori_average_list.append(ori_average)

        if episode < 6:
            print('Episode '+ str(episode) + ' Ave_wait: '+ str(wait_average) + ' AoI: '+ str(aoi_average))
        else:
            print('Episode '+ str(episode) + ' Ave_wait: '+ str(wait_average) + ' AoI: '+ str(aoi_average) + ' Ave_AoI: ' + str(np.mean(aoi_average_list[-5:])))
        #gmma更新
        if episode > 0 and episode % 50 == 0:
            env.gamma += 0.5 * (aoi_average - env.gamma)

        #gat RETRAIN
        #if episode > 0 and episode % 200 == 0 and episode < 1001:
            #RL_wait = DDPG(state_dim=1, action_dim=1)

        episode_time = time.time() - start_time
        episode_time_list.append(episode_time)
        print('Episode Time: '+ str(episode_time))
        print('Average Episode Time: '+ str(np.mean(episode_time_list)))
        if episode % 100 == 0:

            np.savetxt(filepath1, aoi_average_list)
            np.savetxt(filepath2, reward_average_list)
            np.savetxt(filepath3, gamma_list)
            
            np.savetxt(filepath5, wait_average_list)
            # np.savetxt(filepath4, drop_ratio_list)
            # np.savetxt(filepath6, ori_average_list)
            # np.savetxt(filepath7, actor_loss_average)
            # np.savetxt(filepath8, critic_loss_average)
            if TRANSMIT:
                np.savetxt(filepath10, transmit_average) 
                 


    print('game over')

# 添加设备检查
def check_cuda():
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("No CUDA devices available, using CPU")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MEC-DRL')
    parser.add_argument('--mode', type=str, default='Waiting', help='choose a model: Offload_Only, Waiting')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of DRQN')
    parser.add_argument('--comp_iot', type=float, default=2.5, help='Computing capacity of mobile device')
    parser.add_argument('--comp_edge', type=float, default=41.8, help='Computing capacity of edge device')
    parser.add_argument('--comp_cap_edge', type=float, nargs='+', default=[3,8], help='Computing capacity of edge device')
    parser.add_argument('--comp_density', type=float, default=0.297, help='Computing capacity of edge device')
    parser.add_argument('--num_iot', type=int, default=20, help='The number of mobile devices')
    parser.add_argument('--num_edge', type=int, default=2, help='The number of edge devices')
    parser.add_argument('--num_time', type=float, default=300, help='Time per episode')
    parser.add_argument('--num_episode', type=int, default=501, help='number of episode')
    parser.add_argument('--drop_coefficient', type=float, default=1.5, help='number of episode')
    parser.add_argument('--task_size', type=float, default=30, help='Task size (M)')
    parser.add_argument('--gamma', type=float, default=5, help='gamma for fractional')
    parser.add_argument('--folder', type=str, default='standard', help='The folder name of the process')
    parser.add_argument('--subfolder', type=str, default='test', help='The sub-folder name of the process')
    parser.add_argument('--iot_step', type=int, default=0, help='step per iot')
    parser.add_argument('--wait_time', type=float, default=0, help='Fixed waiting time')
    parser.add_argument('--action_range', type=float, default=3, help='Waiting action range')
    parser.add_argument('--FRACTION', type=int, default=1, help='Have fractional AoI or not')
    
    parser.add_argument('--D3QN_NOISE', type=int, default=1, help='Have D3QN noise or not')
    parser.add_argument('--DDPG_NOISE', type=int, default=0, help='Have DDPG noise or not')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='Using GPU')
    parser.add_argument('--STATE_STILL', type=int, default=0, help='STILL STATE?')
    parser.add_argument('--d3qn_step', type=int, default=10, help='D3QN Step')
    parser.add_argument('--WAIT_EXPLORE', type=int, default=300, help='Wait expoloer or not')
    parser.add_argument('--OFFLOAD_EXPLORE', type=int, default=0, help='Wait expoloer or not')
    parser.add_argument('--FULL_NOISE', type=int, default=1, help='Full noise or not')
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--lr_a_o", type=float, default=0.10, help="Learning rate of actor")
    parser.add_argument("--lr_c_o", type=float, default=0.15, help="Learning rate of critic")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--wait_batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--rnn_input_size", type=int, default=8, help="Input size for GRU")
    parser.add_argument("--rnn_hidden_dim", type=int, default=8, help="Hidden size for GRU")
    parser.add_argument("--evaluate_freq", type=float, default=5e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--wait_lr_decay", type=float, default=10, help="lr decay episodes for updating policy")
    parser.add_argument("--pid", type=int, default=0, help="pid")
    parser.add_argument("--LOGNORMAL", type=int, default=0, help="Using lognormal as processing time or not")
    parser.add_argument("--LOG_VARIANCE", type=float, default=1, help="Lognormal variance")
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")
    parser.add_argument("--user_position", type=float, nargs='+', default=[[20, 30], [50, 80], [10, 70], [60, 10], [90, 60], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90], [100, 100], [110, 110], [120, 120], [130, 130], [140, 140], [150, 150]], help="user position")
    parser.add_argument("--server_positions", type=float, nargs='+', default=[[30, 40], [70, 50]], help="server positions")
    parser.add_argument("--bandwidth", type=float, default=10e6, help="Bandwidth (Hz)")
    parser.add_argument("--model_training", type=int, default=1, help="Model training or not")
    parser.add_argument('--TRANSMIT', type=int, default=0, help='Have transmit or not')
    args = parser.parse_args()

    args.pid = os.getpid()
    # 根据CUDA可用性设置device
    if check_cuda():
        args.cuda = args.cuda  # 保持原来的CUDA设备选择
    else:
        args.cuda = 'cpu'  # 如果没有CUDA设备，强制使用CPU
    
    args.state_dim = args.num_edge
    args.action_dim = args.num_edge+1
    print(args)
    # GENERATE ENVIRONMENT
    setup_seed(args.random_seed)
    if args.TRANSMIT:
        from mec_env_test_transmit import Offload_Transmit
        env = Offload_Transmit(args)
    else:
        env = Offload(args)
    ob_shape = list([env.n_features])
    # GENERATE MULTIPLE CLASSES FOR RL
    RL = list()
    RL_wait = list()
    
    for i in range(args.num_iot): 
        RL.append(DRQN(args))
        if args.mode == 'Waiting':
            from RNN_PPO import PPO          
            RL_wait.append(PPO(args=args, state_dim=args.num_edge, action_dim=1, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=10, eps_clip=0.2, has_continuous_action_space=True))
            WAIT = True
        elif args.mode == 'Offload_Only':
            WAIT = False
        


    
    print(f"Using device: {args.cuda}")

    # TRAIN THE SYSTEM
    train(RL, RL_wait,args)

    
    print('Training Finished')
