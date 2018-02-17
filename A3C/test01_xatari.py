import sys
import os
cdir = os.path.split(os.path.abspath(__file__))[0]
projdir = os.path.abspath(os.path.join(cdir, '..'))
sys.path.insert(0, projdir)
print(projdir)
import gym
from gym import envs
import games
import time
import numpy as np
import argparse
#print(envs.registry.all())
#env = gym.make('PongDeterministic-v3')
#env = gym.make('DkDeterministic-v0')
#env = gym.make('BoxingDeterministic-v3')
#env = gym.make('Pitfall-v3')
#env = gym.make('AirRaid-v3')
#env = gym.make('Mariobro-v0')
#env = gym.make('Superman-v0')
parser = argparse.ArgumentParser(description='Python Rom Training')
parser.add_argument('--rom', default='PongDeterministic-v3', type=str, metavar='GYMROMNAME',
                    help='romanme such as  AirRaidDeterministic-v3 (default: PongDeterministic-v3)')

args = parser.parse_args()

live= {'popeye':12}
score= {'popeye':16}

#Romname = 'Superman-v0'
#romname = 'PongDeterministic-v3'
#romname = 'Superman-v0'
romname =args.rom
print ('romname is', romname)
env = gym.make(romname)
#env = gym.make('Aciddrop-v0')
fn = open ('romdebug.csv', 'w')
#env = gym.make('Boxing-v3')
print  (env.action_space)
print  (env.observation_space)
env.reset()

stats={}
stats2={}
inc = 1 
inc2 = -1 

fn.writelines("cat,time,rom,count\n")
for i_epi in range(2):
    observation = env.reset()
    for i in range(10000):
        env.env._render(mode='human')
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
#        print ('action and reward is', action, reward,done) 
        if i>0:
            oldram = ram 
        ram = env.env._get_ram()
        ram = ram.astype(int)

#        if ram [127]>0:
#            print ("the done is", done)
#            done=1

        if done:
            print("Episode finished after {} timesteps".format(i+1))
#            break
#        print (ram)
        if i>0:
            deltaram = ram - oldram

            for index in np.argwhere(deltaram==inc):
                 ind = int(index[0])
                 if ind not in stats.keys():
                     stats[ind] = 0
                 else:
                     stats[ind] = stats[ind]+1

            for index in np.argwhere(deltaram==inc2):
                 ind = int(index[0])
                 if ind not in stats2.keys():
                     stats2[ind] = 0
                 else:
                     stats2[ind] = stats2[ind]+1
             
#            if (len (np.argwhere(deltaram==10))> 0):
#               print ('increement by 10',np.argwhere(deltaram==10))
#            print (ram[98])
#            if deltaram[98]!=0:
#               print (deltaram)
               #print (ram[deltaram==1])
            print ('reward is ', reward)
            if reward != 0:
                print ('live is ', info)
                print ('done ', done)
                print ('score is ', ((ram[16]>>4)*10+(ram[16]&15))+ ((ram[16-1]>>4)*10+(ram[16-1]&15))*100)
            if i %5==0:
                print (i)
#                sorted_names = sorted(stats, key=lambda x: stats[x])
#                for k in sorted_names:
#                      fn.writelines("stats1,{},{},{}\n".format(i,k, stats[k]))
#                sorted_names = sorted(stats2, key=lambda x: stats2[x])
#                for k in sorted_names:
#                      fn.writelines("stats2,{},{},{}\n".format(i,k, stats2[k]))
                for k in range(len(ram)):
                      fn.writelines("rom,{},{},{},{}\n".format(i,k, int(ram[k]),i_epi))
                      fn.writelines("rom_dec,{},{},{},{},{}_{}\n".format(i,k, (ram[k]>>4)*10+(ram[k]&15),i_epi,'singlesplit',k))
                      if k > 1:
                          tmpint2 = ((ram[k]>>4)*10+(ram[k]&15))+ ((ram[k-1]>>4)*10+(ram[k-1]&15))*100
                          fn.writelines("rom_d2,{},{},{},{},{}_{}\n".format(i,k, tmpint2,i_epi,'singlesplit',k))
                      if k > 2:
                          tmpint = ((ram[k]>>4)*10+(ram[k]&15))+ ((ram[k-1]>>4)*10+(ram[k-1]&15))*100+ ((ram[k-2]>>4)*10+(ram[k-2]&15))*10000
                          fn.writelines("rom_d3,{},{},{},{},{}_{}\n".format(i,k, tmpint,i_epi,'singlesplit',k))
#                      if k>0:
#                          fn.writelines("rom_x16,{},{},{},{},{}_{}\n".format(i,k+len(ram)-1, 256*ram[k-1]+ram[k],i_epi,k-1,k))
                         

#        print (ram)
#        print (ram.shape)
#        print (type(observation))
#        print (observation.shape)
#        print ("\n\n")
time.sleep(0.01)
fn.close()
   
