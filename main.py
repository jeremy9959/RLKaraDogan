import numpy as np
import yaml
from supply_chain import SupplyChain
from q_learning_agent import QLearningAgent
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import save
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_episode_for_plotting(agent, supply_chain, k, theta, time_units_per_episode):
    """
    Run an episode and collect data for plotting.

    :param agent: QLearningAgent instance
    :param supply_chain: SupplyChain instance
    :param k: Shape parameter for gamma distribution
    :param theta: Scale parameter for gamma distribution
    :param time_units_per_episode: Number of time units per episode
    :return: Dictionary containing data for plotting
    """
    episode_data = {
        "time": [],
        "reward": [],
        "inventory_size": [],
        "demand": [],
        "expired_units": [],
        "units_ordered": [],
        "unmet_demand": [],
    }

    for t in range(time_units_per_episode):
        # Generate demand from a gamma distribution
        demand = int(np.random.gamma(k, theta))

        # Get the current state (total inventory)
        state = supply_chain.check_inventory()

        # Choose an action (replenishment order quantity) based on the current state
        action = agent.choose_action(state)

        # Advance time by 1 unit with generated demand and chosen replenishment order
        new_total_inventory, reward = supply_chain.advance_time(
            time_units=1, demand=demand, replenishment_order=action
        )

        # Collect data for plotting
        episode_data["time"].append(t)
        episode_data["reward"].append(reward)
        episode_data["inventory_size"].append(new_total_inventory)
        episode_data["demand"].append(demand)
        episode_data["expired_units"].append(supply_chain.expired_units)
        episode_data["units_ordered"].append(action)
        episode_data["unmet_demand"].append(supply_chain.unmet_demand)

    # Create a plot for inventory size vs time
    p5 = figure(
        title="Inventory Size vs Time",
        x_axis_label="Time",
        y_axis_label="Inventory Size",
    )
    p5.line(
        episode_data["time"],
        episode_data["inventory_size"],
        legend_label="Inventory Size",
        line_width=2,
    )

    # Create histograms for units ordered, units expired, and unmet demand
    p6 = figure(
        title="Units Ordered Histogram",
        x_axis_label="Units Ordered",
        y_axis_label="Frequency",
    )
    hist_units_ordered, edges_units_ordered = np.histogram(
        episode_data["units_ordered"], bins=50
    )
    p6.quad(
        top=hist_units_ordered,
        bottom=0,
        left=edges_units_ordered[:-1],
        right=edges_units_ordered[1:],
        fill_color="green",
        line_color="white",
        alpha=0.7,
    )

    p7 = figure(
        title="Units Expired Histogram",
        x_axis_label="Units Expired",
        y_axis_label="Frequency",
    )
    hist_expired_units, edges_expired_units = np.histogram(
        episode_data["expired_units"], bins=50
    )
    p7.quad(
        top=hist_expired_units,
        bottom=0,
        left=edges_expired_units[:-1],
        right=edges_expired_units[1:],
        fill_color="red",
        line_color="white",
        alpha=0.7,
    )

    p8 = figure(
        title="Unmet Demand Histogram",
        x_axis_label="Unmet Demand",
        y_axis_label="Frequency",
    )
    hist_unmet_demand, edges_unmet_demand = np.histogram(
        episode_data["unmet_demand"], bins=50
    )
    p8.quad(
        top=hist_unmet_demand,
        bottom=0,
        left=edges_unmet_demand[:-1],
        right=edges_unmet_demand[1:],
        fill_color="blue",
        line_color="white",
        alpha=0.7,
    )

    # Create a histogram of the inventory size
    p9 = figure(
        title="Inventory Size Histogram",
        x_axis_label="Inventory Size",
        y_axis_label="Frequency",
    )
    hist_inventory_size, edges_inventory_size = np.histogram(
        episode_data["inventory_size"], bins=50
    )
    p9.quad(
        top=hist_inventory_size,
        bottom=0,
        left=edges_inventory_size[:-1],
        right=edges_inventory_size[1:],
        fill_color="navy",
        line_color="white",
        alpha=0.7,
    )

    # Combine the plots into a single layout
    layout = column(p5, p6, p7, p8, p9)

    # Display the plots
    save(layout)
    show(layout)

    return episode_data


def main():
    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Extract parameters from configuration
    state_size = config["state_size"]
    action_size = config["action_size"]
    k = config["k"]
    theta = config["theta"]
    episodes = config["episodes"]
    time_units_per_episode = config["time_units_per_episode"]
    debug = config["debug"]
    alpha = config["alpha"]
    beta = config["beta"]
    initial_inventory = config["initial_inventory"]
    lead_time = config["lead_time"]
    expiration_time = config["expiration_time"]
    learning_rate = config["learning_rate"]
    learning_rate_decay_parameter = config["learning_rate_decay_parameter"]
    discount_factor = config["discount_factor"]
    exploration_rate = config["exploration_rate"]
    exploration_decay_parameter = config["exploration_decay_parameter"]
    min_exploration_rate = config["min_exploration_rate"]
    logging = config["logging"]
    progress_bar = config["progress_bar"]
    trajectory_file_name = config["trajectory_file"]

    # Open the trajectory file for writing
    trajectory_file = open(trajectory_file_name, "w")

    # Initialize the Q-learning agent
    agent = QLearningAgent(
        state_size,
        action_size,
        learning_rate,
        learning_rate_decay_parameter,
        discount_factor,
        exploration_rate,
        exploration_decay_parameter,
        min_exploration_rate,
    )

    # Track total rewards for each episode
    rewards_per_episode = []

    # Track total inventory for each time unit
    total_inventory_data = []

    # Track the number of units ordered for each inventory level
    inventory_order_data = np.zeros((state_size, action_size))

    # progress bars if lprogress_bar flag set to True in config.yaml
    episode_range = tqdm(range(episodes)) if progress_bar else range(episodes)

    for episode in episode_range:
        # Initialize the supply chain for each episode
        supply_chain = SupplyChain(
            initial_inventory=initial_inventory,
            lead_time=lead_time,
            expiration_time=expiration_time,
            alpha=alpha,
            beta=beta,
        )

        # Initialize total rewards for the episode
        total_rewards = 0

        # Report the initial state if debug is True
        if debug:
            initial_state = supply_chain.report_state()
            print(f"Episode {episode + 1} - Initial State:")
            print(f"  Time: {initial_state['time']}")
            print(f"  Total Inventory: {initial_state['total_inventory']}")
            print(f"  Average Age: {initial_state['average_inventory_age']:.2f}")
            print(f"  Orders: {initial_state['orders']}")
            print(f"  Inventory: {initial_state['inventory']}")
            print(f"  Unmet Demand: {initial_state['unmet_demand']}")
            print(f"  Expired Units: {initial_state['expired_units']}")
            print(f"  Inventory Cost: {initial_state['inventory_cost']:.2f}")
            print("=" * 40)

        for t in range(time_units_per_episode):
            # Generate demand from a gamma distribution
            demand = int(np.random.gamma(k, theta))

            # Get the current state (total inventory)
            state = supply_chain.check_inventory()

            # Choose an action (replenishment order quantity) based on the current state
            action = agent.choose_action(state)

            # Track the number of units ordered for each inventory level
            inventory_order_data[state, action] += 1

            # Advance time by 1 unit with generated demand and chosen replenishment order
            new_total_inventory, reward = supply_chain.advance_time(
                time_units=1, demand=demand, replenishment_order=action
            )

            # Get the next state (new total inventory)
            next_state = new_total_inventory

            trajectory_file.write(f"{state},{action},{reward},{next_state}\n")

            # Update the Q-table
            agent.update_q_table(state, action, reward, next_state)

            # Accumulate the reward
            total_rewards += reward

            # Track the total inventory
            total_inventory_data.append(new_total_inventory)

            # Report the current state if debug is True
            if debug:
                state = supply_chain.report_state()
                print(f"Time {state['time']}:")
                print(f"  Demand: {demand}")
                print(f"  Total Inventory: {state['total_inventory']}")
                print(f"  Average Age: {state['average_inventory_age']:.2f}")
                print(f"  Orders: {state['orders']}")
                print(f"  Inventory: {state['inventory']}")
                print(f"  Unmet Demand: {state['unmet_demand']}")
                print(f"  Expired Units: {state['expired_units']}")
                print(f"  Inventory Cost: {state['inventory_cost']:.2f}")
                print(f"  Reward: {reward:.2f}")
                print(f"  New Total Inventory: {new_total_inventory}")
                print("-" * 40)

        # Decay the exploration rate at the end of each episode
        agent.decay_exploration_rate(episode)
        agent.decay_learning_rate(episode)

        # Track the total rewards for the episode
        rewards_per_episode.append(total_rewards / time_units_per_episode)

        # Report the total rewards and epsilon for the episode if logging flag is True in config.yaml
        if logging:
            print(
                f"Episode {episode + 1} - Total Rewards: {total_rewards/time_units_per_episode:.2f}, Epsilon: {agent.exploration_rate:.4f}, Learning Rate: {agent.learning_rate:.4f}"
            )
            print("=" * 40)

    # Plot total rewards vs episodes
    plot_total_rewards(rewards_per_episode)
    plot_q_table_heatmap(agent.q_table[:50, :])
    run_episode_for_plotting(agent, supply_chain, k, theta, time_units_per_episode)


def plot_total_rewards(rewards_per_episode):
    """
    Plot total rewards vs episodes.

    :param rewards_per_episode: List of total rewards for each episode
    """
    p2 = figure(
        title="Avg Reward vs Episodes",
        x_axis_label="Episodes",
        y_axis_label="Total Rewards",
    )
    p2.line(
        list(range(1, len(rewards_per_episode) + 1)),
        rewards_per_episode,
        legend_label="Total Rewards",
        line_width=2,
    )

    # Display the plot

    show(p2)


def plot_q_table_heatmap(q_table):
    """
    Create a heatmap visualization of the Q-table.

    :param q_table: The Q-table to visualize
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, cmap="viridis", center=0)
    plt.title("Q-table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.tight_layout()
    plt.savefig("q_table_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
