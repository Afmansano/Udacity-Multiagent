import torch
import numpy as np
from collections import deque
from agent import Agent, ReplayMemory


class MultiAgent():

    def __init__(self, env, buffer_size, batch_size, gamma, TAU, lr_actor, lr_critic, weight_decay,
                 actor_layers, critic_layers, n_agents, update_every=1, seed=42):
        
        self.brain_name = env.brain_names[0]
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.env = env

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay
        self.UPDATE_EVERY = update_every
        
        # creates a shared memory
        action_size = env.brains[self.brain_name].vector_action_space_size
        self.memory = ReplayMemory(action_size, buffer_size, batch_size, self.DEVICE, seed)

        agent_params = (env, buffer_size, batch_size, gamma, TAU, lr_actor, lr_critic, weight_decay,
                        actor_layers, critic_layers, self.memory, seed)
        self.agents = [Agent(*agent_params) for _ in range(n_agents)]
        self.time_step = 0


    def reset(self):
        for agent in self.agents:
            agent.reset()


    def act(self, states):
        "act throught all agents"
        actions = [agent.act(state)
                   for agent, state in zip(self.agents, states)]
        return actions

    
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.time_step += 1
        if self.time_step % self.UPDATE_EVERY == 0:
            if len(self.memory) > self.BATCH_SIZE:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    agent.learn(*experiences)


    def train(self, n_episodes, max_iterations):
        scores = []
        last_scores = deque(maxlen=100)

        for episode in range(n_episodes):
            self.reset()
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            episode_scores = np.zeros(self.n_agents)

            for _ in range(max_iterations):
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.step(states, actions, rewards, next_states, dones)
                episode_scores += rewards
                states = next_states

            mean_scores = np.mean(episode_scores)
            scores.append(mean_scores)
            last_scores.append(mean_scores)
            last_scores_mean = np.mean(last_scores)
            print('\rEpisode: \t{} \tScore: \t{:.2f} \tMean Scores: \t{:.2f}'.format(episode, mean_scores, last_scores_mean), end="")  

            if last_scores_mean >= 0.5:
                print('\nEnvironment solved in {:d} episodes!\tMean Scores: {:.2f}'.format(episode, last_scores_mean))
                self.agents[0].save()
                break 

        return scores