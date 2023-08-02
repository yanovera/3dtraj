import numpy as np
import matplotlib.pyplot as plt


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

def GPS_sensor(state, noise_std=10):
    # Extract the x, y, z coordinates from the state
    position = state[:3]
    # Generate uncorrelated additive noise
    noise = np.random.normal(0, noise_std, 3)
    # Add the noise to the position
    measurement = position + noise
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

def kalman_filter(states, measurements, dt, process_noise_std, measurement_noise_std):
    # State transition matrix
    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Observation matrix
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    # Process noise covariance
    Q = np.array([
        [dt**4/4, 0, 0, dt**3/2, 0, 0],
        [0, dt**4/4, 0, 0, dt**3/2, 0],
        [0, 0, dt**4/4, 0, 0, dt**3/2],
        [dt**3/2, 0, 0, dt**2, 0, 0],
        [0, dt**3/2, 0, 0, dt**2, 0],
        [0, 0, dt**3/2, 0, 0, dt**2]
    ]) * process_noise_std**2

    # Measurement noise covariance
    R = np.eye(3) * measurement_noise_std**2

    # Initial state estimate and estimate error covariance
    x = np.concatenate((measurements[0], [0, 0, 0]))  # First GPS measurement and zero initial velocities
    P = np.eye(6) * 1000  # Large initial uncertainty

    # Kalman filter
    estimates = []
    for z in measurements:
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q

        # Update
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        P = (np.eye(6) - K @ H) @ P

        estimates.append(x)

    return np.array(estimates)

def intermittent_kalman_filter(states, measurements, dt, downsample_rate, process_noise_std, measurement_noise_std):
    # State transition matrix
    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Observation matrix
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    # Process noise covariance
    Q = np.array([
        [dt**4/4, 0, 0, dt**3/2, 0, 0],
        [0, dt**4/4, 0, 0, dt**3/2, 0],
        [0, 0, dt**4/4, 0, 0, dt**3/2],
        [dt**3/2, 0, 0, dt**2, 0, 0],
        [0, dt**3/2, 0, 0, dt**2, 0],
        [0, 0, dt**3/2, 0, 0, dt**2]
    ]) * process_noise_std**2

    # Measurement noise covariance
    R = np.eye(3) * measurement_noise_std**2

    # Initial state estimate and estimate error covariance
    x = np.concatenate((measurements[0], [0, 0, 0]))  # First GPS measurement and zero initial velocities
    P = np.eye(6) * 1000  # Large initial uncertainty

    # Kalman filter
    estimates = []
    for i, state in enumerate(states):
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q

        # Update if a measurement is available
        if i % downsample_rate == 0:
            z = measurements[i // downsample_rate]
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            x = x + K @ (z - H @ x)
            P = (np.eye(6) - K @ H) @ P

        estimates.append(x)

    return np.array(estimates)




# Generate the trajectory states
states = generate_trajectory()

# Downsample the states to 10 Hz
dt = 0.1  # s
downsample_rate = int(1/dt)
states_downsampled = states[::downsample_rate]

# Generate the GPS measurements
measurements = np.array([GPS_sensor(state) for state in states_downsampled])

# Run the Kalman filter
# estimates = kalman_filter(states_downsampled, measurements, dt, process_noise_std=1, measurement_noise_std=10)

# Run the intermittent Kalman filter
estimates = intermittent_kalman_filter(states, measurements, dt=0.001, downsample_rate=downsample_rate, process_noise_std=1, measurement_noise_std=10)


# Plot the true and estimated trajectories
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states_downsampled[:, 0], states_downsampled[:, 1], states_downsampled[:, 2], label='True')
ax.plot(estimates[:, 0], estimates[:, 1], estimates[:, 2], label='Estimated')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('True vs Estimated Trajectories')
plt.show()