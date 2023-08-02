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

    # Part 2: Circular motion
    w_rad = np.deg2rad(w)  # Convert angular velocity to rad/s
    r = v / w_rad  # Calculate radius of the circle
    center_x = x_straight[-1]  # The circle's center is now at the end of the straight motion
    center_y = y_straight[-1] - r  # The circle's center is now below the end of the straight motion
    angle = w_rad * t_circular  # The angle covered at time t
    x_circular = center_x + r*np.sin(angle)  # Sinusoidal component represents motion along X
    y_circular = center_y + r*np.cos(angle)  # Cosine component represents motion along Y
    z_circular = z_straight[-1] + v*t_circular

    # Part 3: Horizontal motion
    x_horizontal = np.full_like(t_horizontal, x_circular[-1])  # x-coordinate remains constant
    y_horizontal = y_circular[-1] + v*t_horizontal  # y-coordinate increases linearly with time
    z_horizontal = z_circular[-1] + v*t_horizontal

    # Concatenate the positions
    x = np.concatenate((x_straight, x_circular, x_horizontal))
    y = np.concatenate((y_straight, y_circular, y_horizontal))
    z = np.concatenate((z_straight, z_circular, z_horizontal))

    return x, y, z

def plot_trajectory(x, y, z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Trajectory of the Moving Target')
    plt.show()


# Generate and plot the trajectory
x, y, z = generate_trajectory()
plot_trajectory(x, y, z)