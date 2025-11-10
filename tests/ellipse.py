import matplotlib.pyplot as plt
import numpy as np

# Ellipse parameters
a = 0.25000000027997266  # semi-major axis
b = 0.1250000002099795  # semi-minor axis
center = (0.65954144, -0.27554513)
theta = 0.00850917  # rotation in radians

# Ellipse parameterization
t = np.linspace(0, 2 * np.pi, 400)
# parametric ellipse before rotation/translation
x_ = a * np.cos(t)
y_ = b * np.sin(t)

# Rotation matrix
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

xy_rot = R @ np.vstack((x_, y_))
x = xy_rot[0, :] + center[0]
y = xy_rot[1, :] + center[1]

# Plot set up
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, label="Ellipse")

# Plot the domain boundary
domain_x = [-2, 2, 2, -2, -2]
domain_y = [-2, -2, 2, 2, -2]
ax.plot(domain_x, domain_y, "k-", label="Domain boundary")

# Beautify
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
plt.show()
