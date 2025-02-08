def cumulative_reward(rewards):
    return sum(rewards)

def average_reward(rewards):
    if len(rewards) == 0:
        return 0  # 或者返回None，根据需求决定
    return sum(rewards) / len(rewards)

def max_reward(rewards):
    if len(rewards) == 0:
        return None  # 或者返回一个默认值
    return max(rewards)

# Example usage
if __name__ == "__main__":
    rewards = [1, 2, 3, 4, 5]

    print(f"Cumulative Reward: {cumulative_reward(rewards)}")
    print(f"Average Reward: {average_reward(rewards):.4f}")
    print(f"Max Reward: {max_reward(rewards)}")