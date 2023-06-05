

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    global q_values

    batch_size = dqn.batch_size
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!

    sample = memory.sample(batch_size=batch_size)

    observations = torch.cat(sample[0], dim=0).to(device)
    next_observations = torch.cat(sample[2], dim=0).to(device)

    actions = torch.tensor(sample[1], device=device)
    rewards = torch.tensor(sample[3], device=device)
    done = torch.tensor(sample[4], device=device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.

    predictions = dqn.forward(observations).to(device)

    q_values = torch.gather(predictions, dim=1, index=actions.unsqueeze(dim=1)).to(device)

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    non_terminal_states = torch.cat([state for state in next_observations if state is not None])
    with torch.no_grad():
        target_values = target_dqn.forward(non_terminal_states)

    q_value_targets = rewards.unsqueeze(dim=1) + target_dqn.gamma * (torch.mul(target_values.unsqueeze(dim=1),
                                                                               (1 - done.unsqueeze(dim=1))))
    q_value_targets = q_value_targets.to(device)

    loss = F.mse_loss(q_values, q_value_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
