import math
import random
import copy
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import numpy as np


class Blackjack:
    def __init__(self, player_threshold, dealer_threshold, theta=0.000001):
        self.p_threshold = player_threshold
        self.d_threshold = dealer_threshold
        self.theta = theta
        self.card_value_dict = {}
        self.Q = {}
        self.pi = {}
        for i in range(1, 14):

            if i == 1:
                self.card_value_dict[i] = {}
                self.card_value_dict[i]["usable"] = 11
                self.card_value_dict[i]["unusable"] = 1
            elif i >= 10 and i <= 13:
                self.card_value_dict[i] = 10
            else:
                self.card_value_dict[i] = i

    def get_initial_sum(self, player_card_1, player_card_2):
        """helper function to get the initial sum of cards dealt"""
        if player_card_1 == 1 or player_card_2 == 1:
            usable_ace_present = True
            if player_card_1 == 1 and player_card_2 == 1:
                player_sum = 12
            elif player_card_1 == 1:
                player_sum = self.card_value_dict[player_card_2] + 11
            elif player_card_2 == 1:
                player_sum = self.card_value_dict[player_card_1] + 11
        else:
            usable_ace_present = False
            player_sum = (
                self.card_value_dict[player_card_1]
                + self.card_value_dict[player_card_2]
            )
        return player_sum, usable_ace_present

    def play(self, state, action, stick_threshold):
        """play an episode of blackjack"""
        episode = [(state, action)]
        while state[0] <= 21:
            if action == "hit":
                state = self.hit(state, True)
                action = self.pi.get(
                    state, "hit" if state[0] < stick_threshold else "stick"
                )
            elif action == "stick":
                break
            episode.append((state, action))
        episode.append(state)
        # print('finsl',episode)
        return episode

    def hit(self, state, player):
        """returns next state after hitting"""
        next_card = random.randint(1, 13)
        usable_ace = state[2]
        dealer_open_card = state[1]
        # if next card is not an ace
        if next_card != 1:
            current_sum = self.card_value_dict[next_card] + state[0]
            if current_sum > 21:
                # if sum goes above 21; check if usable ace is present
                # if yes then avoid defeat by counting ace as 1 and rendering it unusable
                if usable_ace:
                    current_sum -= 10
                    usable_ace = not usable_ace
        else:  # if next card is an ace get sum assuming it is usable
            current_sum = self.card_value_dict[next_card]["usable"] + state[0]
            if (
                current_sum <= 21
            ):  # if sum is less than or equals 21 you are good and the ace is usable
                usable_ace = True
            if current_sum > 21:  # if sum is greater than 21 subtract -10
                current_sum -= 10
            if current_sum > 21:  # if sum is still greater than 21
                if (
                    usable_ace
                ):  # check if usable ace is present if yes, subtract 10 again to avoid defeat
                    current_sum -= 10
                    usable_ace = not usable_ace
        return (current_sum, dealer_open_card, usable_ace)

    def dealer_play(self, state, stick_threshold):
        """play an episode of blackjack"""
        episode = [state]
        while state[0] < stick_threshold:
            state = self.hit(state, False)
            episode.append(state)
        return episode

    def run(self):
        """run an episode of blackjack"""
        V = {}
        k = 0
        occurence = {}
        while k < 1000000:
            self.dealer_open_card = random.randint(1, 13)
            if self.dealer_open_card != 1:
                dealer_card = self.card_value_dict[self.dealer_open_card]
            else:
                dealer_card = 1
            initial_player_sum = random.randint(11, 21)
            usable_ace = random.choice([True, False])
            initial_player_state = (
                initial_player_sum,
                dealer_card,
                usable_ace,
            )
            if initial_player_sum == 21:
                action = "stick"
            else:
                action = random.choice(["hit", "stick"])
            episode_player = self.play(initial_player_state, action, self.p_threshold)
            if episode_player[-1][0] > 21:
                reward = -1

            else:
                dealer_other_card = random.randint(1, 13)
                initial_dealer_sum, usable_ace = self.get_initial_sum(
                    dealer_card, dealer_other_card
                )
                initial_dealer_state = (
                    initial_dealer_sum,
                    dealer_card,
                    usable_ace,
                )
                episode_dealer = self.dealer_play(
                    initial_dealer_state, self.d_threshold
                )
                if episode_dealer[-1][0] > 21:
                    reward = 1
                elif episode_dealer[-1][0] > episode_player[-1][0]:
                    reward = -1
                elif episode_dealer[-1][0] < episode_player[-1][0]:
                    reward = 1
                else:
                    reward = 0
            for entry in episode_player[: len(episode_player) - 1]:
                if entry[0][0] <= 21 and entry[0][0] >= 11:
                    occurence[entry] = occurence.get(entry, 0) + 1
                    self.Q[entry] = self.Q.get(entry, 0) + 1 / (occurence[entry]) * (
                        reward - self.Q.get(entry, 0)
                    )
                    if self.Q.get((entry[0], "hit"), 0) >= self.Q.get(
                        (entry[0], "stick"), 0
                    ):
                        self.pi[entry[0]] = "hit"
                    else:
                        self.pi[entry[0]] = "stick"

            k += 1
        self.plot_results()
    def plot_results(self):
        '''helper methods to plot results'''
        x_list_hit = []
        y_list_hit = []
        x_list_stick= []
        y_list_stick = []
        x_list1_hit = []
        y_list1_hit = []
        x_list1_stick = []
        y_list1_stick = []
        z_list = []
        for (c, d, u) in self.pi:
            if u:
                if self.pi[(c, d, u)] == "hit":
                    x_list_hit.append(c)
                    y_list_hit.append(d)
                if self.pi[(c, d, u)] == "stick":
                    x_list_stick.append(c)
                    y_list_stick.append(d)
            else:
                if self.pi[(c, d, u)] == "hit":
                    x_list1_hit.append(c)
                    y_list1_hit.append(d)
                if self.pi[(c, d, u)] == "stick":
                    x_list1_stick.append(c)
                    y_list1_stick.append(d)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
        ax[0].scatter(y_list_hit, x_list_hit)
        ax[0].scatter(y_list_stick, x_list_stick)
        ax[0].set(xlabel="dealer card", ylabel="player sum", title=f"usable ace")
        ax[1].scatter(y_list1_hit, x_list1_hit)
        ax[1].scatter(y_list1_stick, x_list1_stick)
        ax[1].set(xlabel="dealer card", ylabel="player sum", title=f" no usable ace")
        ax[0].grid()
        ax[1].grid()
        fig.savefig(f"blackjack_MC_ES.png")
        plt.show()


blackjack = Blackjack(20, 17)
blackjack.run()
