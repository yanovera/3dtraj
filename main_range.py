import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def generate_trajectory(sampling_rate=1000, v=100, w=3):
    # Time intervals
    t_straight = np.linspace(0, 100, 100*sampling_rate)
    t_circular = np.linspace(0, 270/w, int(270/w*sampling_rate))
    t_horizontal = np.linspace(0, 100, 290*sampling_rate - len(t_straight) - len(t_circular))

    # Part 1: Straight motion
    x_straight = 2000 + v*t_straight
    y_straight = np.full_like(t_straight, 2000)
    z_straight = 1000 + v*t_straight
    vx_straight = v*np.ones_like(t_straight)
    vy_straight = np.zeros_like(t_straight)
    vz_straight = v*np.ones_like(t_straight)

    # Part 2: Circular motion
    w_rad = np.deg2rad(w)  # Convert angular velocity to rad/s
    r = v / w_rad  # Calculate radius of the circle
    center_x = x_straight[-1]  # The circle's center is now at the end of the straight motion
    center_y = y_straight[-1] - r  # The circle's center is now below the end of the straight motion
    angle = w_rad * t_circular  # The angle covered at time t
    x_circular = center_x + r*np.sin(angle)  # Sinusoidal component represents motion along X
    y_circular = center_y + r*np.cos(angle)  # Cosine component represents motion along Y
    z_circular = z_straight[-1] + v*t_circular
    vx_circular = v*np.cos(angle)  # Velocity in x direction is v*cos(theta)
    vy_circular = -v*np.sin(angle)  # Velocity in y direction is -v*sin(theta) (negative because the motion is counterclockwise)
    vz_circular = v*np.ones_like(t_circular)

    # Part 3: Horizontal motion
    x_horizontal = np.full_like(t_horizontal, x_circular[-1])  # x-coordinate remains constant
    y_horizontal = y_circular[-1] + v*t_horizontal  # y-coordinate increases linearly with time
    z_horizontal = z_circular[-1] + v*t_horizontal
    vx_horizontal = np.zeros_like(t_horizontal)
    vy_horizontal = v*np.ones_like(t_horizontal)
    vz_horizontal = v*np.ones_like(t_horizontal)

    # Concatenate the positions and velocities
    x = np.concatenate((x_straight, x_circular, x_horizontal))
    y = np.concatenate((y_straight, y_circular, y_horizontal))
    z = np.concatenate((z_straight, z_circular, z_horizontal))
    vx = np.concatenate((vx_straight, vx_circular, vx_horizontal))
    vy = np.concatenate((vy_straight, vy_circular, vy_horizontal))
    vz = np.concatenate((vz_straight, vz_circular, vz_horizontal))

    # Construct the state vectors
    states = np.vstack((x, y, z, vx, vy, vz)).T

    return states

states = generate_trajectory()

# Constants
dt = 1 / 5  # Time step, given 5Hz rate

# Initialize state and covariance matrices
initial_state = states[0]
initial_covariance = np.diag([100 ** 2, 100 ** 2, 100 ** 2, 10 ** 2, 10 ** 2, 10 ** 2])  # Initial uncertainty

# Motion model matrices (constant velocity model)
F = np.array([
    [1, 0, 0, dt, 0, 0],
    [0, 1, 0, 0, dt, 0],
    [0, 0, 1, 0, 0, dt],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

Q = block_diag(np.eye(3) * (10 * dt) ** 2, np.eye(3) * 10 ** 2)  # Process noise

# Sensor positions
sensor_positions = np.array([
    [20000, 10000, -10000],
    [-5000, -10000, -10000],
    [-5000, 10000, 10000],
    [20000, -10000, 10000]
])

# Measurement noise
R = np.array([
    [10 ** 2, 0],
    [0, 15 ** 2]
])


def ekf_predict(state, covariance):
    """
    EKF prediction step.
    """
    # Predict state
    predicted_state = F @ state

    # Predict covariance
    predicted_covariance = F @ covariance @ F.T + Q

    return predicted_state, predicted_covariance


def ekf_update(predicted_state, predicted_covariance, measurements):
    """
    EKF update step using measurements from all sensors.
    """
    for sensor_pos, measurement in zip(sensor_positions, measurements):
        H_jacobian, h = compute_jacobian_and_h(predicted_state, sensor_pos)

        # Kalman gain
        S = H_jacobian @ predicted_covariance @ H_jacobian.T + R
        K = predicted_covariance @ H_jacobian.T @ np.linalg.inv(S)

        # Update state and covariance
        y = measurement - h  # Innovation
        predicted_state = predicted_state + K @ y
        predicted_covariance = (np.eye(6) - K @ H_jacobian) @ predicted_covariance

    return predicted_state, predicted_covariance


def compute_jacobian_and_h(state, sensor_pos):
    """
    Compute the Jacobian matrix and h function for the measurement model.
    """
    pos_diff = state[:3] - sensor_pos
    range_val = np.linalg.norm(pos_diff)

    h = np.array([range_val, np.dot(state[3:], pos_diff / range_val)])

    H = np.zeros((2, 6))
    H[0, :3] = pos_diff / range_val
    H[1, :3] = -state[3:] / range_val
    H[1, 3:] = pos_diff / range_val

    return H, h


def range_sensor(target_state, sensor_pos, range_std=10, doppler_std=15):
    """
    Calculates the range and doppler of a target from a sensor's position.

    Parameters:
    - target_state: [x, y, z, vx, vy, vz] - state of the target.
    - sensor_pos: [x, y, z] - position of the sensor.
    - range_std: standard deviation of the range noise.
    - doppler_std: standard deviation of the doppler noise.

    Returns:
    - [range, doppler]: measurements with noise.
    """

    # Extract target's position and velocity
    target_pos = target_state[:3]
    target_vel = target_state[3:]

    # Calculate range
    range_vector = np.array(target_pos) - np.array(sensor_pos)
    measured_range = np.linalg.norm(range_vector)

    # Calculate Doppler
    unit_range_vector = range_vector / measured_range
    doppler = np.dot(target_vel, unit_range_vector)

    # Add noise
    measured_range += np.random.normal(0, range_std)
    doppler += np.random.normal(0, doppler_std)

    return [measured_range, doppler]


# EKF estimation over the trajectory with matching length

estimated_states = []
estimated_covariances = []

# Initialize state and covariance
state = initial_state
covariance = initial_covariance

# Iterate over the true states to simulate measurements and apply the EKF
for i, true_state in enumerate(states):
    # Predict
    state, covariance = ekf_predict(state, covariance)

    # If it's time for a sensor measurement (i.e., every 5th step)
    if i % 5 == 0:
        # Simulate measurements
        measurements = [range_sensor(true_state, pos) for pos in sensor_positions]
        # Update
        state, covariance = ekf_update(state, covariance, measurements)

    # Store results
    estimated_states.append(state)
    estimated_covariances.append(covariance)

estimated_states = np.array(estimated_states)
estimated_covariances = np.array(estimated_covariances)

# Plot the estimated trajectory and the true trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2], label='True Trajectory', color='blue')
ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label='Estimated Trajectory',
        color='red', linestyle='dotted')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_zlabel('Z position')
ax.set_title('True vs Estimated Trajectory')
ax.legend()
plt.show()