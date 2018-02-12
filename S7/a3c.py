"""

NOTES
After upgrading pytorch to 2.0, the manual seed + span subprocessing (only choice in 2.7)
cause CUDA Error 3.
check the error issue: https://github.com/pytorch/pytorch/issues/2517
"""
from __future__ import print_function
from collections import deque
import time
import os
import torch
from torch.autograd import Variable
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse
import shutil
from utils import FloatTensor, get_elapsed_time_str, SharedAdam
from envs import create_atari_env
from model import ActorCritic

# Parse program arguments
parser = argparse.ArgumentParser(description='Asynchronous Actor Critic')
parser.add_argument('--savedir', default='/tmp', type=str, metavar='PATH',
                    help='Dir name in which we save checkpoints')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help="If checkpoint available, resume from latest")
parser.add_argument('--no-resume', dest='resume', action='store_false')
parser.set_defaults(resume=True)
parser.add_argument('--play', default='', type=str, metavar='PATH',
                    help='play your modle with path specified')
parser.add_argument('--rom', default='PongDeterministic-v4', type=str, metavar='GYMROMNAME',
                    help='Game ROM, e.g. PongDeterministic-v4 (default)')
args = parser.parse_args()
romname = args.rom

SEED = 1


# noinspection PyShadowingNames
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# noinspection PyShadowingNames
def train(rank, shared_model, optimizer):
    """
    :param rank: worker-ID
    :param shared_model: model to sync between workers
    :param optimizer:
    :return:
    """
    # torch.manual_seed(SEED + rank)
    ac_steps = 20 # The amount of steps before you review
    max_episode_length = 10000 # The game will stop after this amount of time and maybe re run the game?
    gamma = 0.99
    tau = 1.0
    max_grad_norm = 50.0 # Limit the direction of gradient travel within the queue. Anything outside the queue is cut
    checkpoint_n = 20 # To see the model after this many n. Can increase this number if have a shit comp

    env = create_atari_env(romname) # enage game. romname is depending on the game of your choice. 
    env.seed(SEED + rank) # For the problem to occur again? LOOK THIS UP
    state = env.reset()
    # Allow torch to handle pixel data. Don't understrand squeeze. FloatTensor - Tensor is an array, therefore array of float.
    state = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor), requires_grad=False)
    # Selecting model, with this size of input and that kind of output
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    t = 0
    done = True # Starting from a state when gameover is true!
    episodes = 0
    reward_sum = 0
    reward_sum1 = 0
    start_time = time.time()
    best_reward = -999
    isbest = 0
    cx = hx = None
    while True:
        model.load_state_dict(shared_model.state_dict()) # Pull the up to date model from the shared model
        if done:  # need to reset LSTM cell's input
            # the LSTM units need their own output to feed into next step
            # input (hence the name of the kind: recurrent neural nets).
            # At the beginning of an episode, to get things started,
            # we need to allocate some initial values in the required format,
            # i.e. the same size as the output of the layer.
            #
            # see http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
            # for details
            #
            # Optionally, you can remove LSTM to simplify the code
            # Think: what is the possible loss?
            cx = Variable(torch.zeros(1, 256)).type(FloatTensor) # torch.zeros - setting the values to all zeros since there's nothing there yet
            hx = Variable(torch.zeros(1, 256)).type(FloatTensor)
        else:
            cx = Variable(cx.data) # takes the last computed value for the next input
            hx = Variable(hx.data)  # basically this is to detach from previous comp graph

        states = []
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for i in range(ac_steps): # Running through the 20 steps
            t += 1
            v, logit, (hx, cx) = model((state, (hx, cx))) # When you run model, it will return you 4 values -> store those 4 values in v, logit, etc.
            states.append(state)
            prob = F.softmax(logit) # The gradient descent thing
            log_prob = F.log_softmax(logit) # Do it again, a lot to make sure its correct
            entropy = -(log_prob * prob).sum(1, keepdim=True) # To increase diversity of our choice (part of e-greedy?)
            entropies.append(entropy)
		
		# detach - anything compute with pytorch will drag a trail behind it. When get gradient descent, the calculation will race with the result. We do not want the descent to chase it randomly, so we just detach it. !Do not need to modify this function when modify the code.
            action = prob.multinomial().detach()  # detach -- so the backprob will NOT go through multinomial()
            # use the current action as an index to get the
            # corresponding log probability
            log_prob = log_prob.gather(1, action) # allow you to simultenously take probability of many actions.

            action = action.data[0, 0] # Extract the variables out of the integer. Turning it from a torch integer to a "normal" integer
            # Accept what was given by the action, does it things? and the env will return the 4 following; state, reward, done
            # _ is something that we don't care about but since env.step is returning 4 values so we just have to have something to take it.
            state, reward, done, _ = env.step(action) 
            reward_sum += reward 
            reward_sum1 += reward # reason why store reward sum twice just for re-assurance
            done = (done or t >= max_episode_length)
            if done:
                t_ = t
                t = 0
                state = env.reset()
                episodes += 1
                if episodes % 10 == 0:
                    time_str = time.strftime(
                        "%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    print("Time {}, worker-{} episode {} "
                          "mean episode reward {}, "
                          "episode length {}".
                          format(time_str, rank, episodes, reward_sum / 10.0, t_))
                    reward_sum = 0.0

                if episodes % checkpoint_n == 0:
                    ave_reward = reward_sum1 / checkpoint_n
                    if best_reward < ave_reward:
                        isbest = 1
                        best_reward = ave_reward

                    print("Saving checkpoint Time {}, worker-{} episode {} "
                          "mean episode reward {}, "
                          "episode length {} best_reward {}".
                          format(get_elapsed_time_str(), rank, episodes, ave_reward, t_, best_reward))
                    checkpoint_fname = os.path.join(
                        args.savedir,
                        args.rom + '_worker' + str(rank) + '_' + str(episodes))
                    save_checkpoint({'epoch': episodes,
                                     'average_reward': ave_reward,
                                     'time': time.time(),
                                     'state_dict': model.state_dict(),
                                     'optimizer': optimizer.state_dict(),
                                     }, isbest, checkpoint_fname)
                    reward_sum1 = 0.0

            state = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor), requires_grad=False)
            reward = max(min(reward, 1), -1)
            values.append(v)
            log_probs.append(log_prob) # Keep record
            rewards.append(reward)

            if done:
                break

        # We reach here because either
        # i) an episode ends, such as game over
        # ii) we have explored certain steps into the future and now it is
        #     time to look-back and summerise the
        if done:
            R = torch.zeros(1, 1).type(FloatTensor) # If game over, the game over stage receive a reward of 0
        else:
            value, _, _ = model((state, (hx, cx))) # if its not game over, then we will use the model to evaluate the reward
            R = value.data

        values.append(Variable(R))
        critic_loss = 0
        actor_loss = 0
        R = Variable(R) 
        gae = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i] # R - longterm reward
            advantage = R - values[i]  # type: Variable, advantage against the average

            # Compare the actual long-term reward. Note: we are reversing the
            # experience of a complete trajectory. If the full length is 100
            # (time indexes are among 0, 1, 2, ..., 99), and now i=50, that means
            # we have processed all information in steps, 51, 52, ..., 99
            # and R will contain the actual long term reward at time step 51 at
            # the beginning of this step. The above computation injects the reward
            # information in step 50 to R. Now R is the long-term reward at this
            # step.
            #
            # So-called advantage is then the "unexpected gain/loss". It forms the base
            # of evaluating the action taken at this step (50).
            #
            # critic_loss accumulates those "exceptional gain/loss" so that later we will
            # adjust our expectation for each state and reduce future exceptions (to better
            # evaluate actions, say, the advantage agains expectation is only meaningful
            # when the expectation itself is meaningful).
            critic_loss += 0.5 * advantage.pow(2)


            # Generalized Advantage Estimation
            # see https://arxiv.org/abs/1506.02438
            # we can use advantage in the computation of the direction to adjust policy,
            # but the manipulation here improves stability (as claims by the paper).
            #
            # Note advantage implicitly contributes to GAE, since it helps
            # achieve a good estimation of state-values.
            td_error = rewards[i] + gamma * values[i + 1].data - values[i].data
            gae = gae * gamma * tau + td_error

            # log_probs[i] is the log-probability(action-taken). If GAE is great, that
            # means the choice we had made was great, and we want to make the same
            # action decision in future -- make log_probs[i] large. Otherwise,
            # we add log_probs to our regret and will be less likely to take the same
            # action in future.
            #
            # entropy means the variety in a probabilistic distribution,
            # to encourage big entropies is to make more exploration.
            actor_loss -= (Variable(gae) * log_probs[i] + 0.01 * entropies[i])

        optimizer.zero_grad() # Applied the gradient to the parameter (back-propagation will get you good stuff from gradient)
        total_loss = actor_loss + critic_loss * 0.5  # type: Variable
        total_loss.backward()  # error occur, back propagation
        # this is to improve stability
        torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
        ensure_shared_grads(model, shared_model) # Push each updated model to the shared model
        optimizer.step()


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        dirname, _ = os.path.split(filename)
        best_fname = os.path.join(dirname, 'best.tar')
        shutil.copyfile(filename, best_fname)


# noinspection PyShadowingNames
def test(shared_model, render=0):
    env = create_atari_env(args.rom)
    if render == 1:
        env.render()

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    # a quick hack to prevent the agent from stucking
    episode_length = 0
    cx = hx = None
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256).type(FloatTensor), volatile=True)
            hx = Variable(torch.zeros(1, 256).type(FloatTensor), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0).type(FloatTensor), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        # print logit.data.numpy()
        action = prob.max(1, keepdim=True)[1].data.cpu().numpy()

        state, reward, done, _ = env.step(action[0, 0])
        if render:
            env.render()
        done = done or episode_length >= 10000
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # actions.append(action[0, 0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".
                  format(get_elapsed_time_str(), reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(60)
        state = torch.from_numpy(state)


if __name__ == '__main__':
    env = create_atari_env(args.rom)
    # torch.manual_seed(SEED)
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    # print (shared_model.conv1._parameters['weight'].data.is_cuda)
    optimizer = SharedAdam(shared_model.parameters(), lr=0.0001)
    optimizer.share_memory()

    if args.play:
        if os.path.isfile(args.play):
            print("=> loading checkpoint '{}'".format(args.play))
            checkpoint = torch.load(args.play)
            #            args.start_epoch = checkpoint['epoch']
            #            best_prec1 = checkpoint['best_prec1']
            shared_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.play))

        test(shared_model, render=1)  # let it play the game
        exit(0)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #            args.start_epoch = checkpoint['epoch']
            #            best_prec1 = checkpoint['best_prec1']
            shared_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    mp.set_start_method('spawn')
    processes = []
    p = mp.Process(target=test, args=(shared_model, 0))
    p.start()
    processes.append(p)
	
	# This loop start the processes
    for rank in range(0, 3): # This loop how many agent we shall run simultaneously
        print("Starting {}".format(rank))
        p = mp.Process(target=train, args=(rank, shared_model, optimizer))
        p.start() # Start point
        processes.append(p)

    for p in processes:
        p.join()
