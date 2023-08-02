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

def radar_sensor(state, noise_std=np.array([0.01, 0.01, 10, 1])):
    # Extract the position and velocity from the state
    position = state[:3]
    velocity = state[3:]

    # Calculate the azimuth, elevation, range, and Doppler
    x, y, z = position
    vx, vy, vz = velocity
    R = np.linalg.norm(position)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    D = (vx*x + vy*y + vz*z) / R

    # Generate uncorrelated additive noise
    noise = np.random.normal(0, noise_std, 4)

    # Add the noise to the measurements
    measurement = np.array([theta, phi, R, D]) + noise

    return measurement

def plot_trajectory(x, y, z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Trajectory of the Moving Target')
    plt.show()

def ekf_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P

def ekf_update(x, P, z, h, H, R):
    y = z - h(x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P

def intermittent_extended_kalman_filter(states, measurements, dt, downsample_rate, process_noise_std, measurement_noise_std):
    # State transition function and its Jacobian
    def f(x):
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F @ x

    F = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Observation function and its Jacobian
    def h(x):
        x, y, z, vx, vy, vz = x
        R = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        D = (vx*x + vy*y + vz*z) / R
        return np.array([theta, phi, R, D])

    def H(x):
        x, y, z, vx, vy, vz = x
        R = np.sqrt(x**2 + y**2 + z**2)
        R3 = R**3
        H = np.zeros((4, 6))
        H[0, :] = np.array([-y/R**2, x/R**2, 0, 0, 0, 0])
        H[1, :] = np.array([-x*z/R3, -y*z/R3, np.sqrt(x**2 + y**2)/R3, 0, 0, 0])
        H[2, :] = np.array([x/R, y/R, z/R, 0, 0, 0])
        H[3, :] = np.array([-vx*x/R + x*(vx*x + vy*y + vz*z)/R3, -vy*x/R + y*(vx*x + vy*y + vz*z)/R3,
                            -vz*x/R + z*(vx*x + vy*y + vz*z)/R3, x/R, y/R, z/R])
        return H

    # Process noise covariance
    Q = block_diag(np.zeros((3, 3)), np.eye(3)*process_noise_std**2)

    # Measurement noise covariance
    R = np.diag(measurement_noise_std**2)

    # Initial state estimate and estimate error covariance
    x = np.concatenate((measurements[0, 2]*np.array([np.cos(measurements[0, 0])*np.cos(measurements[0, 1]),
                                                     np.sin(measurements[0, 0])*np.cos(measurements[0, 1]),
                                                     np.sin(measurements[0, 1])]),
                        [0, 0, 0]))  # First Radar measurement converted back to Cartesian coordinates and zero initial velocities
    P = np.eye(6) * 1000  # Large initial uncertainty

    # Intermittent extended Kalman filter
    estimates = []
    for i, state in enumerate(states):
        # Predict
        x, P = ekf_predict(x, P, F, Q)

        # Update if a measurement is available
        if i % downsample_rate == 0:
            z = measurements[i // downsample_rate]
            x, P = ekf_update(x, P, z, h, H(x), R)

        estimates.append(x)

    return np.array(estimates)

# Generate the trajectory states
states = generate_trajectory()

# Downsample the states to 10 Hz
dt = 0.1  # s
downsample_rate = int(1/dt)
states_downsampled = states[::downsample_rate]

# Generate the Radar measurements
measurements = np.array([radar_sensor(state) for state in states_downsampled])

# Run the intermittent extended Kalman filter
estimates = intermittent_extended_kalman_filter(states, measurements, dt=0.001, downsample_rate=downsample_rate, process_noise_std=1, measurement_noise_std=np.array([0.01, 0.01, 10, 1]))

# Plot the true and estimated trajectories
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2], label='True')
ax.plot(estimates[:, 0], estimates[:, 1], estimates[:, 2], label='Estimated')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('True vs Estimated Trajectories')
plt.show()