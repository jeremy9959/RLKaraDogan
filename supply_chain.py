import collections


class SupplyChain:
    def __init__(
        self, initial_inventory=0, lead_time=1, expiration_time=10, alpha=1, beta=1
    ):
        """
        Initialize the supply chain with an initial inventory level, lead time, expiration time, and cost weights.

        :param initial_inventory: Initial inventory level
        :param lead_time: Lead time for orders to arrive
        :param expiration_time: Time after which units expire
        :param alpha: Weight for unmet demand in cost calculation
        :param beta: Weight for expired units in cost calculation
        """
        if expiration_time <= lead_time:
            raise ValueError("Expiration time must be greater than lead time.")

        self.inventory = collections.deque()
        self.orders = collections.deque()
        self.lead_time = lead_time
        self.expiration_time = expiration_time
        self.alpha = alpha
        self.beta = beta
        self.time = 0
        self.unmet_demand = 0
        self.expired_units = 0

        # Initialize inventory with initial_inventory units
        if initial_inventory > 0:
            self.inventory.append((initial_inventory, self.time))

    def place_order(self, quantity):
        """
        Place an order to the supplier.

        :param quantity: Quantity to order
        """
        arrival_time = self.time + self.lead_time
        self.orders.append((quantity, arrival_time))
        # Removed print statement

    def receive_order(self):
        """
        Receive orders from the supplier if they have arrived.
        """
        while self.orders and self.orders[0][1] <= self.time:
            quantity, _ = self.orders.popleft()
            self.inventory.append((quantity, self.time))
            # Removed print statement

    def discard_expired_units(self):
        """
        Discard expired units from the inventory.
        """
        while (
            self.inventory and self.inventory[0][1] <= self.time - self.expiration_time
        ):
            quantity, _ = self.inventory.popleft()
            self.expired_units += quantity

    def check_inventory(self):
        """
        Check the current inventory level.

        :return: Current inventory level
        """
        return sum(quantity for quantity, _ in self.inventory)

    def satisfy_demand(self, demand):
        """
        Satisfy customer demand from stocks. Any excess demand is lost.

        :param demand: Quantity of demand to satisfy
        """
        satisfied = 0
        while demand > 0 and self.inventory:
            quantity, time_added = self.inventory.popleft()
            if quantity > demand:
                satisfied += demand
                self.inventory.appendleft((quantity - demand, time_added))
                demand = 0
            else:
                satisfied += quantity
                demand -= quantity
        self.unmet_demand += demand
        return demand

    def place_replenishment_order(self, order_quantity):
        """
        Place a replenishment order based on the specified order quantity.

        :param order_quantity: Quantity to order
        """
        if order_quantity > 0:
            self.place_order(order_quantity)

    def advance_time(self, time_units=1, demand=0, replenishment_order=0):
        """
        Advance the time by a specified number of time units.

        :param time_units: Number of time units to advance
        :param demand: Customer demand to satisfy
        :param replenishment_order: Quantity to order for replenishment
        :return: Tuple containing the new total inventory and the reward for the period
        """
        for _ in range(time_units):
            self.unmet_demand = 0
            self.expired_units = 0
            self.time += 1
            self.receive_order()
            self.satisfy_demand(demand)
            self.discard_expired_units()
            self.place_replenishment_order(replenishment_order)

        new_total_inventory = self.check_inventory()
        reward = -self.calculate_inventory_cost()
        return new_total_inventory, reward

    def get_average_inventory_age(self):
        """
        Get the average age of the current inventory.

        :return: Average age of the current inventory
        """
        total_quantity = self.check_inventory()
        if total_quantity == 0:
            return 0
        total_age = sum(
            (self.time - time_added) * quantity
            for quantity, time_added in self.inventory
        )
        return total_age / total_quantity

    def report_state(self):
        """
        Report the current state of the supply chain.

        :return: A dictionary containing the current state of the supply chain
        """
        state = {
            "time": self.time,
            "total_inventory": self.check_inventory(),
            "average_inventory_age": self.get_average_inventory_age(),
            "orders": list(self.orders),
            "inventory": list(self.inventory),
            "unmet_demand": self.unmet_demand,
            "expired_units": self.expired_units,
            "inventory_cost": self.calculate_inventory_cost(),
        }
        return state

    def calculate_inventory_cost(self):
        """
        Calculate the inventory cost based on unmet demand and expired units for the current period.

        :return: Inventory cost for the current period
        """
        return self.alpha * self.unmet_demand + self.beta * self.expired_units
