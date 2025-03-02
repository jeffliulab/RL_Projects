"""
This code is based on the idea of Example 5.3 in Sutton & Barto's "Reinforcement Learning: An Introduction."
It implements an "abstract" Blackjack environment (with an infinite deck) and uses Monte Carlo ES (Exploring Starts)
to learn the optimal policy. It does not contain any "hard-coded" or "manually specified" strategy rules,
but rather relies on the Monte Carlo method to converge automatically after a sufficient number of training episodes.

Latest modifications:
1) Uses Monte Carlo ES + epsilon-greedy in subsequent decisions to learn the optimal Blackjack policy;
2) Trains on a sufficiently large number of episodes (default 5,000,000) and fixes the random seed for reproducibility.

Key points:
1. State representation: s = (player_sum, dealer_upcard, usable_ace)
   - player_sum ∈ [12..21]: sums below 12 are not included in the state space, because the player will always hit.
   - dealer_upcard ∈ [1..10]: the dealer's face-up card (1 represents Ace).
   - usable_ace ∈ {True, False}: whether the player has a usable Ace (i.e., counted as 11 without busting).

2. Actions:
   - 0 = Stick
   - 1 = Hit

3. Exploring Starts:
   - Randomly choose an initial (s, a) from all possible states and actions, then follow the current greedy policy
     w.r.t. Q for subsequent steps.
   - This ensures that every (s, a) has a nonzero probability of being sampled.

4. Environment logic:
   - If the player chooses Hit, a card is drawn from the infinite deck (1..9 each with probability 1/13, 10 with probability 4/13),
     and the player's sum and "usable_ace" status are updated. If the sum exceeds 21, the player immediately loses (reward = -1).
   - If the player chooses Stick, the dealer draws cards (dealer_play) until the total is at least 17 or the dealer busts, then
     compares totals with the player to determine the outcome (+1 / -1 / 0).

5. Training and convergence:
   - By default, we run 5,000,000 episodes to ensure sufficient sampling of Q(s,a) and the policy for convergence.
   - With a consistent environment setup (infinite deck, dealer must hit below 17, no doubling/splitting, etc.),
     this method will converge, after enough simulations, to an optimal policy and value function very close to Figure 5.2 in the book.
   - If you want further smoothing or faster convergence, you can increase the number of episodes or use a small epsilon
     in epsilon-greedy. This example does keep a small epsilon in subsequent steps.

6. Visualization:
   - A 2×2 figure layout is produced:
     - (row=0) usable ace, (row=1) no usable ace
     - (col=0) policy, (col=1) value function
   - The policy subplots use colored blocks to distinguish 0=Stick and 1=Hit, while the value subplots use a color gradient
     to show V(s). With sufficient convergence, the resulting plots closely resemble the examples in the book.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from tqdm import trange

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)

STICK = 0
HIT = 1
ACTIONS = [STICK, HIT]

def draw_card():
    """
    Draw a card from an infinite deck (1..9 each with probability 1/13, and 10 with probability 4/13).
    """
    c = np.random.randint(1, 14)  # 1..13
    return min(c, 10)

def hand_value(sum_without_ace, ace_count):
    """
    Given sum_without_ace (the total of non-A cards) and ace_count (the number of Aces, each counted as 1),
    if at least one Ace can be counted as 11 without busting, treat one Ace as 11 (total += 10).
    Returns (total, usable_ace).
    """
    total = sum_without_ace + ace_count
    usable = False
    if ace_count > 0 and total + 10 <= 21:
        total += 10
        usable = True
    return total, usable

def player_hit(player_sum, usable_ace):
    """
    Player takes a hit. Returns (new_sum, new_usable_ace, bust).
    """
    card = draw_card()
    # Convert (player_sum, usable_ace) back to (sum_without_ace, ace_count)
    if usable_ace:
        ace_count = 1
        sum_without_ace = player_sum - 11
    else:
        ace_count = 0
        sum_without_ace = player_sum

    if card == 1:
        ace_count += 1
    else:
        sum_without_ace += card

    new_sum, new_usable = hand_value(sum_without_ace, ace_count)
    bust = (new_sum > 21)
    return new_sum, new_usable, bust

def dealer_play(dealer_upcard):
    """
    The dealer first draws a hidden card, then continues to hit until >=17 or bust.
    Returns (dealer_sum, dealer_usable_ace).
    """
    second = draw_card()
    sum_without_ace = 0
    ace_count = 0

    # Dealer's face-up card
    if dealer_upcard == 1:
        ace_count += 1
    else:
        sum_without_ace += dealer_upcard

    # Dealer's hidden card
    if second == 1:
        ace_count += 1
    else:
        sum_without_ace += second

    dealer_sum, dealer_usable = hand_value(sum_without_ace, ace_count)
    while dealer_sum < 17:
        c = draw_card()
        if c == 1:
            ace_count += 1
        else:
            sum_without_ace += c
        dealer_sum, dealer_usable = hand_value(sum_without_ace, ace_count)
        if dealer_sum > 21:
            break

    return dealer_sum, dealer_usable

def step(state, action):
    """
    One step in the environment:
    state = (player_sum, dealer_up, usable_ace)
    action = 0(STICK) or 1(HIT)
    Returns (next_state, reward, done).
    """
    player_sum, dealer_up, player_usable = state

    if action == HIT:
        new_sum, new_usable, bust = player_hit(player_sum, player_usable)
        if bust:
            return None, -1, True
        else:
            return (new_sum, dealer_up, new_usable), 0, False
    else:
        # STICK => dealer's turn
        dealer_sum, _ = dealer_play(dealer_up)
        if dealer_sum > 21:
            return None, +1, True
        else:
            if player_sum > dealer_sum:
                return None, +1, True
            elif player_sum < dealer_sum:
                return None, -1, True
            else:
                return None, 0, True

def random_state_action():
    """
    Exploring Starts: randomly pick (s,a) from [12..21]×[1..10]×{False,True}.
    """
    player_sum = np.random.randint(12, 22)
    dealer_up = np.random.randint(1, 11)
    ace = np.random.choice([False, True])
    action = np.random.choice(ACTIONS)
    return (player_sum, dealer_up, ace), action

def generate_episode_es_epsilon(Q, epsilon=0.05):
    """
    Generate one episode:
    1) Exploring Starts: random (s,a) for the first step
    2) Then follow an epsilon-greedy(Q) policy until termination
    Returns a list of (state, action, reward).
    """
    episode = []
    state, action = random_state_action()
    done = False

    while True:
        next_state, reward, done = step(state, action)
        episode.append((state, action, reward))
        if done:
            break

        # epsilon-greedy for subsequent actions
        state = next_state
        if np.random.rand() < epsilon:
            action = np.random.choice(ACTIONS)
        else:
            action = np.argmax(Q[state])

    return episode

def mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05):
    """
    Monte Carlo ES + epsilon-greedy for subsequent decisions.
    Default episodes = 5,000,000.
    """
    Q = defaultdict(lambda: np.zeros(2))
    returns = defaultdict(list)

    for _ in trange(num_episodes, desc="Training"):
        episode = generate_episode_es_epsilon(Q, epsilon)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G += r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns[(s_t, a_t)].append(G)
                Q[s_t][a_t] = np.mean(returns[(s_t, a_t)])

    policy = {}
    for s, q_vals in Q.items():
        policy[s] = np.argmax(q_vals)

    return Q, policy

def get_value_function(Q):
    """
    V(s) = max_a Q(s,a)
    """
    V = {}
    for s, q_vals in Q.items():
        V[s] = np.max(q_vals)
    return V

# ---------- 2×2 Visualization ----------
def plot_policy_value_2x2(policy, V):
    """
    Creates a 2×2 figure:
      Top row: usable ace
        Left: policy   Right: value
      Bottom row: no usable ace
        Left: policy   Right: value

    Policy subplot: color map for 0=Stick, 1=Hit
    Value subplot: gradient (e.g., 'viridis') for V(s)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Two rows (usable_ace=True/False), two columns (policy/value)
    for row_idx, ace in enumerate([True, False]):
        # --- Policy (col=0) ---
        ax_policy = axes[row_idx, 0]
        x_vals = range(1, 11)   # dealer up
        y_vals = range(12, 22)  # player sum
        policy_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)

        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                policy_grid[j, i] = policy.get(s, 0)

        # Discrete cmap, e.g. 'Blues', with 0 and 1
        im1 = ax_policy.imshow(
            policy_grid,
            origin='lower',
            extent=[1, 10, 12, 21],
            cmap=plt.cm.Blues,
            vmin=0, vmax=1,
            aspect='auto'
        )
        ax_policy.set_xlabel("Dealer showing")
        ax_policy.set_ylabel("Player sum")
        ax_policy.set_title(f"Policy (usable_ace={ace})\n0=Stick, 1=Hit")

        # --- Value (col=1) ---
        ax_value = axes[row_idx, 1]
        value_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)
        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                value_grid[j, i] = V.get(s, 0.0)

        im2 = ax_value.imshow(
            value_grid,
            origin='lower',
            extent=[1, 10, 12, 21],
            cmap='viridis',
            aspect='auto'
        )
        ax_value.set_xlabel("Dealer showing")
        ax_value.set_ylabel("Player sum")
        ax_value.set_title(f"Value (usable_ace={ace})")

        fig.colorbar(im1, ax=ax_policy, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax_value, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    # 1) Train 5,000,000 episodes with epsilon=0.05
    Q, policy = mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05)
    V = get_value_function(Q)

    # 2) 2×2 figure
    plot_policy_value_2x2(policy, V)
