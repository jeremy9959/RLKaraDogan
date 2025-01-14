import numpy as np
import yaml
from supply_chain import SupplyChain
from q_learning_agent import QLearningAgent
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import save

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
    max_inventory = config["max_inventory"]
    learning_rate = config["learning_rate"]
    discount_factor = config["discount_factor"]
    exploration_rate = config["exploration_rate"]
    exploration_decay = config["exploration_decay"]
    min_exploration_rate = config["min_exploration_rate"]
    
    # Initialize the Q-learning agent
    agent = QLearningAgent(state_size, action_size, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration_rate)
    
    # Track total rewards for each episode
    rewards_per_episode = []
    
    # Track total inventory for each time unit
    total_inventory_data = []
    
    # Track the number of units ordered for each inventory level
    inventory_order_data = np.zeros((state_size, action_size))
    
    for episode in range(episodes):
        # Initialize the supply chain for each episode
        supply_chain = SupplyChain(initial_inventory=initial_inventory, lead_time=lead_time, expiration_time=expiration_time, alpha=alpha, beta=beta, max_inventory=max_inventory)
        
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
            new_total_inventory, reward = supply_chain.advance_time(time_units=1, demand=demand, replenishment_order=action)
            
            # Get the next state (new total inventory)
            next_state = new_total_inventory
            
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
        agent.decay_exploration_rate()
        
        # Track the total rewards for the episode
        rewards_per_episode.append(total_rewards)
        
        # Report the total rewards and epsilon for the episode
        print(f"Episode {episode + 1} - Total Rewards: {total_rewards:.2f}, Epsilon: {agent.exploration_rate:.4f}")
        print("=" * 40)
    
    # Run one additional episode and collect data for plotting
    additional_episode_data = {
        "time": [],
        "inventory_size": [],
        "demand": [],
        "expired_units": [],
        "units_ordered": [],
        "unmet_demand": []
    }
    
    supply_chain = SupplyChain(initial_inventory=initial_inventory, lead_time=lead_time, expiration_time=expiration_time, alpha=alpha, beta=beta, max_inventory=max_inventory)
    for t in range(time_units_per_episode):
        # Generate demand from a gamma distribution
        demand = int(np.random.gamma(k, theta))
        
        # Get the current state (total inventory)
        state = supply_chain.check_inventory()
        
        # Choose an action (replenishment order quantity) based on the current state
        action = agent.choose_action(state)
        
        # Advance time by 1 unit with generated demand and chosen replenishment order
        new_total_inventory, reward = supply_chain.advance_time(time_units=1, demand=demand, replenishment_order=action)
        
        # Collect data for plotting
        additional_episode_data["time"].append(t)
        additional_episode_data["inventory_size"].append(new_total_inventory)
        additional_episode_data["demand"].append(demand)
        additional_episode_data["expired_units"].append(supply_chain.expired_units)
        additional_episode_data["units_ordered"].append(action)
        additional_episode_data["unmet_demand"].append(supply_chain.unmet_demand)
    
    # Create a plot for total rewards vs episodes
    p2 = figure(title="Total Rewards vs Episodes", x_axis_label="Episodes", y_axis_label="Total Rewards")
    p2.line(list(range(1, episodes + 1)), rewards_per_episode, legend_label="Total Rewards", line_width=2)
    
    # Create a histogram of the total inventory
    p3 = figure(title="Total Inventory Histogram", x_axis_label="Total Inventory", y_axis_label="Frequency")
    hist, edges = np.histogram(total_inventory_data, bins=50)
    p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.7)
    
    # Create a plot for inventory size vs time
    p5 = figure(title="Inventory Size vs Time", x_axis_label="Time", y_axis_label="Inventory Size")
    p5.line(additional_episode_data["time"], additional_episode_data["inventory_size"], legend_label="Inventory Size", line_width=2)
    
    # Create histograms for units ordered, units expired, and unmet demand
    p6 = figure(title="Units Ordered Histogram", x_axis_label="Units Ordered", y_axis_label="Frequency")
    hist_units_ordered, edges_units_ordered = np.histogram(additional_episode_data["units_ordered"], bins=50)
    p6.quad(top=hist_units_ordered, bottom=0, left=edges_units_ordered[:-1], right=edges_units_ordered[1:], fill_color="green", line_color="white", alpha=0.7)
    
    p7 = figure(title="Units Expired Histogram", x_axis_label="Units Expired", y_axis_label="Frequency")
    hist_expired_units, edges_expired_units = np.histogram(additional_episode_data["expired_units"], bins=50)
    p7.quad(top=hist_expired_units, bottom=0, left=edges_expired_units[:-1], right=edges_expired_units[1:], fill_color="red", line_color="white", alpha=0.7)
    
    p8 = figure(title="Unmet Demand Histogram", x_axis_label="Unmet Demand", y_axis_label="Frequency")
    hist_unmet_demand, edges_unmet_demand = np.histogram(additional_episode_data["unmet_demand"], bins=50)
    p8.quad(top=hist_unmet_demand, bottom=0, left=edges_unmet_demand[:-1], right=edges_unmet_demand[1:], fill_color="blue", line_color="white", alpha=0.7)
    
    # Combine the plots into a single layout
    layout = column(p2, p3, p5, p6, p7, p8)
    
    # Display the plots
    save(layout)
    show(layout)
    
    # Report the most likely actions for each state
    print("Most Likely Actions for Each State:")
    for state in range(state_size):
        max_value = np.max(agent.q_table[state])
        max_actions = np.where(agent.q_table[state] == max_value)[0]
        print(f"State {state}: Most Likely Actions = {max_actions}")

if __name__ == "__main__":
    main()
