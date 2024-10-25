import pyglet
from pyglet.window import key
import math
from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Config Variables
USE_AI = True  # Set to True for AI control, False for user control
SHOW_VISUALIZER = False  # Set to True to show neural network visualizer, False to disable

LAP_REWARD = 10000
CRASH_PUNISHMENT = 500
GATE_REWARD = 100  # Base reward for the first gate
TIME_PUNISHMENT = 1

EXPLOSION_COLOR_DEFAULT = (255, 165, 0)  # Orange
EXPLOSION_COLOR_LAP = (128, 0, 128)      # Purple

# Maximum multiplier to cap reward growth
MAX_MULTIPLIER = 16

# Interface for drawable objects
class Drawable(ABC):
    @abstractmethod
    def draw(self):
        pass

# Interface for updatable objects
class Updatable(ABC):
    @abstractmethod
    def update(self, dt):
        pass

# Input handler class
class InputHandler:
    def __init__(self):
        self.keys = key.KeyStateHandler()

    def get_keys(self):
        return self.keys

# AI Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64

        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = self.build_model().to(self.device)
        self.target_net = self.build_model().to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.update_counter = 0  # To update target network periodically

        # For capturing activations
        self.activations = {}

    def build_model(self):
        # Adjust input layer size based on state_size (6 sensors + speed + angle)
        model = nn.Sequential(
            nn.Linear(self.state_size, 16),  # Hidden layer
            nn.ReLU(),
            nn.Linear(16, self.action_size)  # Output layer
        )
        return model

    def forward(self, x):
        # Custom forward method to capture activations
        activations = {}

        x = x.to(self.device)
        x = x.float()

        # Input to hidden layer
        x = self.policy_net[0](x)
        activations['layer1'] = x.detach().cpu().numpy()
        x = self.policy_net[1](x)

        # Hidden to output layer
        x = self.policy_net[2](x)
        activations['output'] = x.detach().cpu().numpy()

        # Store activations
        self.activations = activations

        return x

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            # Random action
            action = random.randrange(self.action_size)
            # When taking random action, capture zero activations
            self.activations = {
                'layer1': np.zeros((1, 16)),
                'output': np.zeros((1, self.action_size))
            }
            return action
        else:
            # Use the custom forward method to capture activations
            with torch.no_grad():
                q_values = self.forward(state_tensor)
            action = torch.argmax(q_values[0]).item()
            return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            target_q_values = rewards + self.gamma * torch.max(self.target_net(next_states), dim=1)[0] * (1 - dones)

        target_q_values = target_q_values.unsqueeze(1)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_network()

# Neural Network Visualizer Class (Simplified)
class NeuralNetworkVisualizer(Drawable):
    def __init__(self, agent, x, y):
        self.agent = agent
        self.x = x
        self.y = y

        # Simplified input layer: sensors, speed, position, angle
        self.layer_nodes = {
            'input': [
                'Sensor Front-Left', 
                'Sensor Front', 
                'Sensor Front-Right', 
                'Sensor Left', 
                'Sensor Right', 
                'Sensor Back',
                'Speed',
                'Angle'
            ],
            'output': ['No Action', 'Accelerate', 'Decelerate', 'Turn Left', 'Turn Right']
        }
        self.positions = {}
        self.calculate_positions()

        # Store connections for coloring
        self.connections = []

    def calculate_positions(self):
        # Calculate positions for nodes
        layer_spacing = 200  # Spacing between input and output layers
        node_spacing_input = 50  # Spacing for input nodes
        node_spacing_output = 50  # Spacing for output nodes

        # Input layer positions
        input_x = self.x
        total_height_input = node_spacing_input * (len(self.layer_nodes['input']) - 1)
        input_y_start = self.y - total_height_input / 2
        self.positions['input'] = []
        for i, node in enumerate(self.layer_nodes['input']):
            y = input_y_start + i * node_spacing_input
            self.positions['input'].append((input_x, y, node))

        # Output layer positions
        output_x = self.x + layer_spacing
        total_height_output = node_spacing_output * (len(self.layer_nodes['output']) - 1)
        output_y_start = self.y - total_height_output / 2
        self.positions['output'] = []
        for i, node in enumerate(self.layer_nodes['output']):
            y = output_y_start + i * node_spacing_output
            self.positions['output'].append((output_x, y, node))

    def draw(self):
        # Draw nodes and labels
        for layer_name, nodes in self.positions.items():
            for x, y, label_text in nodes:
                circle = pyglet.shapes.Circle(
                    x, y, 15, color=(255, 255, 255)
                )
                circle.draw()
                label = pyglet.text.Label(
                    label_text,
                    font_name='Arial',
                    font_size=10,  # Adjusted font size for clarity
                    x=x, y=y - 25,
                    anchor_x='center', anchor_y='top',
                    color=(255, 255, 255, 255)
                )
                label.draw()

        # Draw connections with colors based on effective weights
        self.update_connections()
        for connection in self.connections:
            x1, y1, x2, y2, weight = connection
            # Map weight to color
            color = self.weight_to_color(weight)
            line = pyglet.shapes.Line(
                x1, y1, x2, y2, width=2, color=color
            )
            line.draw()

    def update_connections(self):
        # Clear previous connections
        self.connections = []

        # Get effective weights from input to output
        state_dict = self.agent.policy_net.state_dict()

        # Input to hidden layer weights
        w1 = state_dict['0.weight'].cpu().numpy()  # Shape: [16, 8]

        # Hidden to output layer weights
        w2 = state_dict['2.weight'].cpu().numpy()  # Shape: [5, 16]

        # Compute effective weights: output = w2 * relu(w1 * input)
        # For visualization, we'll compute w2 @ w1
        effective_weights = np.dot(w2, w1)  # Shape: [5,8]

        input_nodes = self.positions['input']
        output_nodes = self.positions['output']

        for i, (x1, y1, _) in enumerate(input_nodes):
            for j, (x2, y2, _) in enumerate(output_nodes):
                weight = effective_weights[j][i]
                self.connections.append((x1, y1, x2, y2, weight))

    def weight_to_color(self, weight):
        # Normalize weight to range [0, 1] based on expected weight range
        # Adjust max_weight and min_weight as needed based on training
        max_weight = 5.0
        min_weight = -5.0
        norm_weight = (weight - min_weight) / (max_weight - min_weight)
        norm_weight = np.clip(norm_weight, 0, 1)

        # Map to color gradient from blue (negative) to red (positive)
        r = int(norm_weight * 255)
        b = int((1 - norm_weight) * 255)
        g = 0
        return (r, g, b)

# Track class (Updated with lap tracking and programmatic reward multipliers)
class Track(Drawable):
    def __init__(self):
        self.batch = pyglet.graphics.Batch()

        self.outer_radius = 170  # Increased from 160 to 170
        self.inner_radius = 80   # Decreased from 90 to 80

        outer_radius = self.outer_radius
        inner_radius = self.inner_radius

        # Background
        self.background = pyglet.shapes.Rectangle(
            0, 0, 1200, 900, color=(0, 0, 0), batch=self.batch
        )

        # Road (green oval)
        self.left_outer_circle = pyglet.shapes.Circle(
            300, 450, outer_radius, color=(0, 255, 0), batch=self.batch
        )
        self.right_outer_circle = pyglet.shapes.Circle(
            900, 450, outer_radius, color=(0, 255, 0), batch=self.batch
        )
        self.track_rect = pyglet.shapes.Rectangle(
            300, 450 - outer_radius, 600, 2 * outer_radius, color=(0, 255, 0), batch=self.batch
        )

        # Inner area (black)
        self.left_inner_circle = pyglet.shapes.Circle(
            300, 450, inner_radius, color=(0, 0, 0), batch=self.batch
        )
        self.right_inner_circle = pyglet.shapes.Circle(
            900, 450, inner_radius, color=(0, 0, 0), batch=self.batch
        )
        self.inner_track_rect = pyglet.shapes.Rectangle(
            300, 450 - inner_radius, 600, 2 * inner_radius, color=(0, 0, 0), batch=self.batch
        )

        # Finish line
        self.finish_line = pyglet.shapes.Line(
            300, 450 - inner_radius, 300, 450 + inner_radius, width=2, color=(255, 255, 255), batch=self.batch
        )

        # Initialize starting gate index before creating reward gates
        self.starting_gate_index = 4  # Gate where a lap starts
        self.active_gate_index = self.starting_gate_index
        self.lap_counter = 0  # Initialize lap counter

        # Reward Gates
        self.reward_gates = self.create_reward_gates(20, self.starting_gate_index)

        # Set gate colors
        for gate in self.reward_gates:
            gate.active = False
            gate.color = (255, 255, 0)  # Yellow for inactive gates

        # Activate the starting gate
        self.reward_gates[self.active_gate_index].active = True
        self.reward_gates[self.active_gate_index].color = (0, 0, 255)  # Blue for active gate

    def create_reward_gates(self, num_gates, starting_gate_index):
        gates = []

        # Calculate lengths of track segments
        straight_length = 600  # Length of straight sections
        curve_length = math.pi * self.outer_radius  # Length of curved sections (half-circumference)
        total_length = 2 * straight_length + 2 * curve_length

        # Segment length
        segment_length = total_length / num_gates

        positions = []
        current_length = 0

        # Create positions for gates
        for i in range(num_gates):
            # Get position along the track
            pos = self.get_position_along_track(current_length)
            positions.append(pos)
            current_length += segment_length

        # Create gates without assigning multipliers yet
        for pos in positions:
            x_outer, y_outer, x_inner, y_inner, angle = pos
            gate = pyglet.shapes.Line(
                x_outer, y_outer, x_inner, y_inner, width=2, color=(255, 255, 0), batch=self.batch
            )
            gate.angle = angle
            gate.active = False  # Inactive by default
            gates.append(gate)

        # Assign reward multipliers starting from the starting_gate_index
        current_multiplier = 1
        for i in range(num_gates):
            gate_index = (starting_gate_index + i) % num_gates
            gate = gates[gate_index]
            gate.reward_multiplier = current_multiplier
            current_multiplier = min(current_multiplier * 2, MAX_MULTIPLIER)  # Cap the multiplier

        return gates

    def get_position_along_track(self, distance):
        # Calculate lengths of track segments
        straight_length = 600  # Length of straight sections
        curve_length = math.pi * self.outer_radius  # Length of curved sections (half-circumference)
        total_length = 2 * straight_length + 2 * curve_length

        distance = distance % total_length  # Loop around the track

        if distance < straight_length:
            # Top straight section (from right to left)
            ratio = distance / straight_length
            x_outer = 900 - ratio * 600
            y_outer = 450 + self.outer_radius
            x_inner = 900 - ratio * 600
            y_inner = 450 + self.inner_radius
            angle = 180  # Facing left
        elif distance < straight_length + curve_length:
            # Left curve (from top to bottom)
            ratio = (distance - straight_length) / curve_length
            theta = math.pi / 2 + ratio * math.pi  # From 90° to 270°
            x_outer = 300 + self.outer_radius * math.cos(theta)
            y_outer = 450 + self.outer_radius * math.sin(theta)
            x_inner = 300 + self.inner_radius * math.cos(theta)
            y_inner = 450 + self.inner_radius * math.sin(theta)
            angle = (math.degrees(theta) + 90) % 360
        elif distance < 2 * straight_length + curve_length:
            # Bottom straight section (from left to right)
            ratio = (distance - straight_length - curve_length) / straight_length
            x_outer = 300 + ratio * 600
            y_outer = 450 - self.outer_radius
            x_inner = 300 + ratio * 600
            y_inner = 450 - self.inner_radius
            angle = 0  # Facing right
        else:
            # Right curve (from bottom to top)
            ratio = (distance - 2 * straight_length - curve_length) / curve_length
            theta = 3 * math.pi / 2 + ratio * math.pi  # From 270° to 450°
            x_outer = 900 + self.outer_radius * math.cos(theta)
            y_outer = 450 + self.outer_radius * math.sin(theta)
            x_inner = 900 + self.inner_radius * math.cos(theta)
            y_inner = 450 + self.inner_radius * math.sin(theta)
            angle = (math.degrees(theta) + 90) % 360

        return (x_outer, y_outer, x_inner, y_inner, angle)

    def draw(self):
        self.batch.draw()

    def is_on_road(self, corners):
        # Check if all corners are on the road
        for x, y in corners:
            if not self.point_is_on_road(x, y):
                return False
        return True

    def point_is_on_road(self, x, y):
        # The road is defined as the area between the inner and outer boundaries
        left_center = (300, 450)
        right_center = (900, 450)

        if x <= 300:
            dist = math.hypot(x - left_center[0], y - left_center[1])
            return self.inner_radius <= dist <= self.outer_radius
        elif x >= 900:
            dist = math.hypot(x - right_center[0], y - right_center[1])
            return self.inner_radius <= dist <= self.outer_radius
        else:
            distance_from_center_y = abs(y - 450)
            return self.inner_radius <= distance_from_center_y <= self.outer_radius

    def check_reward_gate(self, x, y, previous_x, previous_y):
        gate = self.reward_gates[self.active_gate_index]

        # Check for intersection
        car_start = (previous_x, previous_y)
        car_end = (x, y)
        gate_start = (gate.x, gate.y)
        gate_end = (gate.x2, gate.y2)

        if self.lines_intersect(car_start, car_end, gate_start, gate_end):
            # Compute car's movement vector
            car_dx = x - previous_x
            car_dy = y - previous_y

            # Compute gate's normal vector
            gate_dx = gate.x2 - gate.x
            gate_dy = gate.y2 - gate.y
            gate_normal_x = -gate_dy
            gate_normal_y = gate_dx

            # Compute dot product
            dot = car_dx * gate_normal_x + car_dy * gate_normal_y

            if dot < 0:
                # Car is moving in the correct direction
                # Deactivate current gate
                gate.active = False
                gate.color = (255, 255, 0)  # Yellow

                # Move to next gate
                self.active_gate_index = (self.active_gate_index + 1) % len(self.reward_gates)
                next_gate = self.reward_gates[self.active_gate_index]
                next_gate.active = True
                next_gate.color = (0, 0, 255)  # Blue

                # Check if a lap is completed
                lap_completed = (self.active_gate_index == self.starting_gate_index)
                if lap_completed:
                    self.lap_counter += 1

                return (True, lap_completed, gate.reward_multiplier)
        return (False, False, 1)  # Default multiplier is 1 if no gate passed

    def lines_intersect(self, a1, a2, b1, b2):
        # Return True if line segments a1a2 and b1b2 intersect
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        A = a1
        B = a2
        C = b1
        D = b2
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def angle_to_next_gate(self, x, y, heading):
        # Calculate the angle difference to the next gate
        gate = self.reward_gates[self.active_gate_index]
        gate_x = (gate.x + gate.x2) / 2
        gate_y = (gate.y + gate.y2) / 2
        angle_to_gate = math.degrees(math.atan2(gate_y - y, gate_x - x))
        angle_diff = (angle_to_gate - heading + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        return angle_diff

    def reset_reward_gates(self):
        for gate in self.reward_gates:
            gate.active = False
            gate.color = (255, 255, 0)  # Yellow
        self.active_gate_index = self.starting_gate_index  # Reset to the starting gate
        self.reward_gates[self.active_gate_index].active = True
        self.reward_gates[self.active_gate_index].color = (0, 0, 255)  # Blue

# Car class (Updated to handle lap completion, purple explosion, and dynamic gate rewards)
class Car(Drawable, Updatable):
    def __init__(self, x, y, input_handler, track):
        self.start_x = x
        self.start_y = y
        self.heading = 180  # Facing left down the track
        self.x = x
        self.y = y
        self.speed = 0
        self.max_speed = 200
        self.acceleration = 200
        self.rotation_speed = 100
        self.input_handler = input_handler
        self.track = track
        self.alive = True
        self.explosion_timer = 0

        # For AI
        self.agent = None
        self.state = None
        self.total_reward = 0

        # Explosion attributes
        self.explosion_color = EXPLOSION_COLOR_DEFAULT  # Default orange
        self.exploded_after_laps = False  # Flag to ensure explosion after 3 laps only once

        # Create the car sprite
        self.sprite = pyglet.shapes.Rectangle(
            self.x, self.y, 20, 40, color=(255, 0, 0)
        )
        self.sprite.anchor_x = self.sprite.width / 2
        self.sprite.anchor_y = self.sprite.height / 2

        # Sensor visualization
        self.sensors = []

        # Store previous position
        self.previous_x = x
        self.previous_y = y

        # Add a reference to the GameWindow to update generation counter
        self.game_window = None  # Will be set later

    def reset_position(self):
        self.x = self.start_x
        self.y = self.start_y
        self.heading = 180  # Reset heading to face down the track
        self.speed = 0
        self.alive = True
        self.explosion_timer = 0
        self.total_reward = 0
        self.state = None
        self.sensors.clear()
        self.previous_x = self.x
        self.previous_y = self.y

        # Reset reward gates when the car resets
        self.track.reset_reward_gates()

        # Reset explosion attributes
        self.explosion_color = EXPLOSION_COLOR_DEFAULT
        self.exploded_after_laps = False

        # Increment generation counter
        if self.game_window:
            self.game_window.generation_counter += 1

    def update(self, dt):
        if not self.alive:
            # Update explosion timer
            self.explosion_timer -= dt
            if self.explosion_timer <= 0:
                self.reset_position()
            return

        # Store previous position
        self.previous_x = self.x
        self.previous_y = self.y

        if USE_AI and self.agent:
            # AI Control
            next_state = self.get_state()
            if self.state is None:
                self.state = next_state
            action = self.agent.act(self.state)
            self.perform_action(action, dt)
            reward = -TIME_PUNISHMENT  # Default small penalty for each step

            # Check for collision with track boundaries using car's edges
            if not self.track.is_on_road(self.get_corners()):
                self.alive = False
                self.explosion_timer = 1.5  # Explosion lasts for 1.5 seconds
                reward = -CRASH_PUNISHMENT  # Large penalty for crashing
                done = True
            else:
                # Check for passing reward gates
                passed_gate, lap_completed, gate_multiplier = self.track.check_reward_gate(
                    self.x, self.y, self.previous_x, self.previous_y
                )
                if passed_gate:
                    reward = GATE_REWARD * gate_multiplier  # Reward for passing a gate with multiplier
                    if lap_completed:
                        reward += LAP_REWARD  # Reward for completing a lap
                        # Trigger explosion after 3 laps
                        if self.track.lap_counter >= 3 and not self.exploded_after_laps:
                            self.alive = False
                            self.explosion_timer = 1.5  # Explosion duration
                            self.explosion_color = EXPLOSION_COLOR_LAP  # Change to purple
                            self.exploded_after_laps = True

                done = False

            self.agent.remember(self.state, action, reward, next_state, done)
            self.agent.replay()
            self.total_reward += reward
            self.state = next_state
        else:
            # User Control
            keys = self.input_handler.get_keys()
            # Store previous position
            self.previous_x = self.x
            self.previous_y = self.y

            # Rotate the car
            if keys[key.A]:
                self.heading += self.rotation_speed * dt
            if keys[key.D]:
                self.heading -= self.rotation_speed * dt

            # Accelerate or decelerate the car
            if keys[key.W]:
                self.speed += self.acceleration * dt
            elif keys[key.S]:
                self.speed -= self.acceleration * dt
            else:
                # Apply friction when no key is pressed
                self.speed *= 0.98

            # Limit the speed
            self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

            # Update the car's position
            rad = math.radians(self.heading)
            self.x += math.cos(rad) * self.speed * dt
            self.y += math.sin(rad) * self.speed * dt

            # Check for collision with track boundaries using car's edges
            if not self.track.is_on_road(self.get_corners()):
                self.alive = False
                self.explosion_timer = 1.5  # Explosion lasts for 1.5 seconds

        # Update the sprite's position and rotation
        self.sprite.x = self.x
        self.sprite.y = self.y
        self.sprite.rotation = -self.heading + 90  # Adjust for sprite orientation

    def perform_action(self, action, dt):
        # Actions: 0=No Action, 1=Accelerate, 2=Decelerate, 3=Steer Left, 4=Steer Right
        if action == 1:
            self.speed += self.acceleration * dt
        elif action == 2:
            self.speed -= self.acceleration * dt
        elif action == 3:
            self.heading += self.rotation_speed * dt
        elif action == 4:
            self.heading -= self.rotation_speed * dt
        else:
            # Apply friction when no action is taken
            self.speed *= 0.98

        # Limit the speed
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

        # Update the car's position
        rad = math.radians(self.heading)
        self.x += math.cos(rad) * self.speed * dt
        self.y += math.sin(rad) * self.speed * dt

    def get_state(self):
        # State consists of distances to boundaries and heading to next gate
        distances = self.get_sensor_readings()
        speed = self.speed / self.max_speed  # Normalize speed
        angle_to_gate = self.track.angle_to_next_gate(self.x, self.y, self.heading)
        state = distances + [speed, angle_to_gate / 180]  # Normalize angle
        return state

    def get_sensor_readings(self):
        # Simulate sensors in front-left, front, front-right, left, right, and back directions
        sensor_angles = [-90, -45, 0, 45, 90, 180]  # Relative to car's heading
        readings = []
        self.sensors.clear()  # Clear previous sensor lines
        for angle in sensor_angles:
            distance, end_x, end_y = self.cast_ray(angle)
            readings.append(distance / 200)  # Normalize distances
            # Add sensor line for visualization
            self.sensors.append(((self.x, self.y), (end_x, end_y)))
        return readings

    def cast_ray(self, relative_angle):
        # Cast a ray in a given direction and return the distance to the boundary
        angle = math.radians(self.heading + relative_angle)
        step = 5
        max_distance = 200
        distance = 0
        x = self.x
        y = self.y
        while distance < max_distance:
            x += math.cos(angle) * step
            y += math.sin(angle) * step
            distance += step
            if not self.track.point_is_on_road(x, y):
                break
        return distance, x, y

    def draw(self):
        if self.alive:
            self.sprite.draw()
            # Draw sensor lines
            for sensor in self.sensors:
                start, end = sensor
                line = pyglet.shapes.Line(
                    start[0], start[1], end[0], end[1], width=1, color=(0, 255, 255)
                )
                line.opacity = 150  # Semi-transparent
                line.draw()
        else:
            # Draw explosion with dynamic color
            explosion = pyglet.shapes.Circle(self.x, self.y, 30, color=self.explosion_color)
            explosion.opacity = int(255 * (self.explosion_timer / 1.5))  # Fade out
            explosion.draw()

    def get_corners(self):
        # Calculate the positions of the car's corners for collision detection
        rad = math.radians(self.heading)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)

        half_width = self.sprite.width / 2
        half_height = self.sprite.height / 2

        # Corners relative to the car's center
        corners = [
            (-half_width, -half_height),
            (-half_width, half_height),
            (half_width, half_height),
            (half_width, -half_height),
        ]

        # Rotate and translate corners
        world_corners = []
        for dx, dy in corners:
            world_x = self.x + dx * cos_rad - dy * sin_rad
            world_y = self.y + dx * sin_rad + dy * cos_rad
            world_corners.append((world_x, world_y))

        return world_corners

# Game class (Updated to display lap count and handle gate rewards)
class GameWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(1200, 900, "Oval Race Track")  # Increased window size
        self.input_handler = InputHandler()
        if not USE_AI:
            self.push_handlers(self.input_handler.get_keys())

        self.track = Track()
        # Start the car on the top straight section within the green oval, facing left
        self.car = Car(
            480,
            450 + (self.track.inner_radius + self.track.outer_radius) / 2,
            self.input_handler,
            self.track
        )
        self.car.game_window = self  # Set reference to update generation counter

        if USE_AI:
            state_size = 8  # Updated state_size: 6 sensors + speed + angle to gate
            action_size = 5  # Five possible actions
            self.car.agent = DQNAgent(state_size, action_size)

        self.updatables = [self.car]
        self.drawables = [self.track, self.car]

        # Initialize generation counter
        self.generation_counter = 1

        # Create neural network visualizer
        if USE_AI and SHOW_VISUALIZER:
            self.nn_visualizer = NeuralNetworkVisualizer(self.car.agent, x=800, y=800)
            self.drawables.append(self.nn_visualizer)

        # Schedule the update function
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    def on_draw(self):
        self.clear()
        for drawable in self.drawables:
            drawable.draw()
        # Display generation counter
        generation_label = pyglet.text.Label(
            f"Generation: {self.generation_counter}",
            font_name='Arial',
            font_size=16,
            x=10, y=self.height - 20,
            anchor_x='left', anchor_y='center',
            color=(255, 255, 255, 255)
        )
        generation_label.draw()
        if USE_AI:
            # Display AI's total reward
            reward_label = pyglet.text.Label(
                f"Total Reward: {int(self.car.total_reward)}",
                font_name='Arial',
                font_size=16,
                x=10, y=self.height - 50,
                anchor_x='left', anchor_y='center',
                color=(255, 255, 255, 255)
            )
            reward_label.draw()
            # Display lap count
            lap_label = pyglet.text.Label(
                f"Laps Completed: {self.track.lap_counter}",
                font_name='Arial',
                font_size=16,
                x=10, y=self.height - 80,
                anchor_x='left', anchor_y='center',
                color=(255, 255, 255, 255)
            )
            lap_label.draw()

    def update(self, dt):
        for updatable in self.updatables:
            updatable.update(dt)

# Run the application
if __name__ == "__main__":
    window = GameWindow()
    pyglet.app.run()
