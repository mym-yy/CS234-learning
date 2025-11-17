### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    ############################
    # YOUR IMPLEMENTATION HERE #
    num_states, num_actions = R.shape
    backup_val = R[state, action]
    expected_val = 0.
    for i in range(num_states):
        expected_val += T[state, action, i] * V[i]
    backup_val += gamma * expected_val
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        V_old = value_function.copy()
        max_delta = 0

        for s in range(num_states):
            a = policy[s]
            V_new_s = bellman_backup(s, a, R, T, gamma, value_function)
            delta = np.abs(V_new_s - V_old[s])
            value_function[s] = V_new_s
            max_delta = max(delta, max_delta)

        if max_delta < tol:
            break
    ############################
    return value_function

def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states),policy下的状态价值函数
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(num_states):
        q_values = np.zeros(num_actions)

        for a in range(num_actions):
            q_values[a] = bellman_backup(s, a, R, T, gamma, V_policy)

        new_policy[s] = np.argmax(q_values)
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        #策略评估
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        #策略改进
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        if np.all(new_policy == policy):
            break

        policy = new_policy.copy()
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        V_old = value_function.copy()
        max_delta = 0

        #找到每个状态的最优值函数才能完成迭代
        for s in range(num_states):
            q_values = np.zeros(num_actions)

            for a in range(num_actions):
                #计算每个动作的Q值
                q_values[a] = bellman_backup(s, a, R, T, gamma, value_function)

            V_new_s = np.max(q_values)
            delta = np.abs(V_new_s - V_old[s])
            value_function[s] = V_new_s
            max_delta = max(delta, max_delta)

        if max_delta < tol:
            break

    #根据最终的值函数提取策略
    #最终的策略是每个状态下选择使得Q值最大的动作
    for s in range(num_states):
        q_values = np.zeros(num_actions)

        for a in range(num_actions):
            q_values[a] = bellman_backup(s, a, R, T, gamma, value_function)

        policy[s] = np.argmax(q_values)
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
