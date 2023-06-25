import math
import random
import copy
import time
import matplotlib.pyplot as plt

class Gambler():
    def __init__(self,max_capital,ph,immediate_reward,loss_reward,gain_reward,theta=0.000001):
        self.max_capital=max_capital
        self.ph=ph
        self.gain_reward=gain_reward
        self.immediate_reward=immediate_reward
        self.loss_reward=loss_reward
        self.V={}
        self.pi={}
        self.N_states=N_states=list(range(0,self.max_capital+1))
        self.theta=theta
        self.p={}
    def dynamics(self):
        ''' get the state transition probabilities'''
    
        N_non_terminal_states=self.N_states[1:-1]
        print(N_non_terminal_states)
        for i in N_non_terminal_states:
            for j in range(0,min(i,self.max_capital-i)+1):
                self.p[(i,j)]={}
                result_capital_win,result_capital_lose,r_win,r_lose=self.get_resulting_capital(i,j)
                self.p[(i,j)][(result_capital_win,r_win)]=self.ph
                self.p[(i,j)][(result_capital_lose,r_lose)]=1-self.ph

    def get_resulting_capital(self,current_capital,current_action):
        ''' get the resulting state(capital) and reward for each given current state and action'''
        # if win
        capital_win=current_capital+current_action
        # if lose
        capital_lose=current_capital-current_action
        r_win=self.gain_reward if capital_win==self.max_capital else self.immediate_reward
        if r_win==1:
          print(r_win,current_action,current_capital)

          # assert 1==2
        r_lose=self.loss_reward if capital_lose==0 else self.immediate_reward
        return capital_win,capital_lose,r_win,r_lose
    def value_iteration(self):
         for i in self.N_states:
            self.V[i]=2 if i!=0 and i!=self.max_capital else 0
         iterate=True
         while iterate:
             delta=0
             for s in self.N_states[1:-1]:
                q_max=-float('inf')
                q_values=[]
                for a in range(0,min(s,self.max_capital-s)+1):
                    q=0
                    for (s_,r) in self.p[(s,a)]:
                        q+=self.p[(s,a)][(s_,r)]*(r+self.V[s_])

                    if q>q_max:
                        q_max=q
                        a_max=a
                        print('current value at ',s,q_max,a_max)
                v=self.V[s]
                q_max_val=[k for k in q_values if k==q_max]
                self.V[s]=q_max
                self.pi[s]=a_max
                # print(f'iteration {it} state {s} value {self.V[s]} action {self.pi[s]} ')
                delta=max(delta,abs(self.V[s]-v))
                # print(delta)
             if delta<self.theta:
                print(delta)
                iterate=False
    def gamble(self):
        self.dynamics()
        self.value_iteration()
        self.plot()
    def plot(self):
        s=list(self.pi.keys())
        a=list(self.pi.values())
        plt.figure()
        plt.xlabel('current capital', fontsize=20)
        plt.ylabel('capital gambled', fontsize=20)
        
        plt.plot(s,a)
        
        plt.figure()
        plt.xlabel('current_capital',fontsize=20)
        plt.ylabel('value',fontsize=20)
        plt.plot(self.V.keys(),self.V.values())
        plt.show()


gambler=Gambler(100,0.4,0,-10,1)
gambler.gamble()












