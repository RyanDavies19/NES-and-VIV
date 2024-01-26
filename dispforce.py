import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Read
df1 = pd.read_csv('/Users/argy/Documents/GitHub/MoorDyn_testing/MooringTest/hanging_springCpy_Line1.csv', delimiter=r'\s+', engine='python')
df2 = pd.read_csv('/Users/argy/Documents/GitHub/MoorDyn_testing/MooringTest/hanging_springCpy_Line2.csv', delimiter=r'\s+', engine='python')

# Skip the row with units and reset
df1.columns = ['Time', 'Node0px', 'Node0py', 'Node0pz', 'Node1px', 'Node1py', 'Node1pz']
df2.columns = ['Time', 'Node0px', 'Node0py', 'Node0pz', 'Node1px', 'Node1py', 'Node1pz']
df1 = df1.iloc[1:].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
df2 = df2.iloc[1:].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
# Define time_values as the time column from your DataFrame
time_values = df1['Time'] if len(df1) >= len(df2) else df2['Time']
# Setup the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize scatter plots
scatters = [ax.scatter(df1['Node0px'][0], df1['Node0py'][0], df1['Node0pz'][0], label='Node0 (DF1)'),
            ax.scatter(df1['Node1px'][0], df1['Node1py'][0], df1['Node1pz'][0], label='Node1 (DF1)'),
            ax.scatter(df2['Node0px'][0], df2['Node0py'][0], df2['Node0pz'][0], label='Node0 (DF2)', marker='^'),
            ax.scatter(df2['Node1px'][0], df2['Node1py'][0], df2['Node1pz'][0], label='Node1 (DF2)', marker='^')]

# Setting axis labels and legend
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.legend()

# Setting the animation update function
def update(frame_number):
    for scatter, df, node_prefix in zip(scatters, [df1, df1, df2, df2], ['Node0', 'Node1', 'Node0', 'Node1']):
        scatter._offsets3d = (df[node_prefix + 'px'][:frame_number],
                              df[node_prefix + 'py'][:frame_number],
                              df[node_prefix + 'pz'][:frame_number])
    return scatters

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=max(len(df1), len(df2)), interval=500, blit=False)


plt.show()

def calculate_midpoint(point_a, point_b):
    # Calculate the midpoint between two points in 3D
    return [(a + b) / 2 for a, b in zip(point_a, point_b)]


# Calculate the midpoint between 'Node0' from df1 and 'Node0' from df2
midpoint_node0 = calculate_midpoint(df1[['Node0px', 'Node0py', 'Node0pz']].iloc[0],
                                    df2[['Node0px', 'Node0py', 'Node0pz']].iloc[0])


def calculate_displacement(df, node_prefix='Node1'):
    # Extract the initial position of the node
    initial_position = df[[f'{node_prefix}px', f'{node_prefix}py', f'{node_prefix}pz']].iloc[0].values

    # Calculate the displacement as the Euclidean distance from the initial position
    displacement = df.apply(lambda row: np.linalg.norm(
        row[[f'{node_prefix}px', f'{node_prefix}py', f'{node_prefix}pz']].values - initial_position), axis=1)

    return displacement


def calculate_midpoint(point_a, point_b):
    # Calculate the midpoint between two points in 3D
    return [(a + b) / 2 for a, b in zip(point_a, point_b)]


# Calculate the midpoint between 'Node0' from df1 and 'Node0' from df2
midpoint_node0 = calculate_midpoint(df1[['Node0px', 'Node0py', 'Node0pz']].iloc[0],
                                    df2[['Node0px', 'Node0py', 'Node0pz']].iloc[0])

#  calculate_displacement
def calculate_displacement_from_midpoint(df, node_prefix='Node1', initial_position=midpoint_node0):
    # Calculate the displacement as the Euclidean distance from the midpoint
    displacement = df.apply(lambda row: np.linalg.norm(
        row[[f'{node_prefix}px', f'{node_prefix}py', f'{node_prefix}pz']].values - initial_position), axis=1)

    return displacement


df1_displacement_from_midpoint = 1*calculate_displacement_from_midpoint(df1, 'Node1', midpoint_node0)

df1_displacement = calculate_displacement(df1, 'Node1')

df = pd.read_csv('/Users/argy/Documents/GitHub/MoorDyn_testing/MooringTest/hanging_springCpy.csv', delimiter=r'\s+', engine='python')

numeric_cols = df.columns.drop('Time')
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop any rows
df.dropna(inplace=True)

# Reset
df.reset_index(drop=True, inplace=True)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df1_displacement_from_midpoint,-df['Point1FZ'], label='Displacement of Node1')
plt.xlabel('Displacement (m)')
plt.ylabel('Force (N)')
plt.title('Displacement of Node1 vs Force')
plt.legend()
plt.grid(True)
plt.show()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time_values,df1_displacement_from_midpoint, label='Displacement of Node1')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement of Node1 vs Force')
plt.legend()
plt.grid(True)
plt.show()

displacement = df1_displacement_from_midpoint
force = -df['Point1FZ']

# Try fitting polynomials of different orders
for order in range(1, 5):
    coefs = np.polyfit(displacement, force, order)
    polynomial = np.poly1d(coefs)
    force_pred = polynomial(displacement)
    r2 = r2_score(force, force_pred)
    print(f"Order: {order}, Coefficients: {coefs}, R^2: {r2}")

    # Plot the fit
    plt.figure()
    plt.scatter(displacement, force, label='Original Data')
    displacement_line = np.linspace(min(displacement), max(displacement), 1000)
    force_line = polynomial(displacement_line)
    plt.plot(displacement_line, force_line, label=f'Order {order} Fit')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Force (N)')
    plt.title(f'Displacement vs Force (Polynomial Order {order})')
    plt.legend()
    plt.grid(True)
    plt.show()
