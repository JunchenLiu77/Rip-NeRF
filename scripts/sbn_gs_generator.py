import numpy as np
import open3d as o3d
import argparse


def fibonacci_sphere(samples=10):
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(samples)
    theta = 2 * np.pi * i / phi
    z = (2 * i + 1) / samples - 1
    radius = np.sqrt(1 - z * z)
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    return np.column_stack((x, y, z))


def advance_spherical_blue_noise(
    particles, max_displacement, iterations, convergence_threshold=1e-3
):
    # source: https://github.com/nouvadam/spherical-blue-noise
    for _ in range(iterations):
        updated_particles = []
        total_displacement = 0
        for i, particle in enumerate(particles):
            other_particles = np.concatenate((particles[:i], particles[i + 1 :]))
            displacement = np.zeros(3)
            for other_particle in other_particles:
                diff = particle - other_particle
                diff_norm = np.linalg.norm(diff)
                if diff_norm > 0:
                    displacement += diff / (diff_norm**3)
            displacement /= np.linalg.norm(displacement)
            updated_particle = particle + max_displacement * displacement
            updated_particle /= np.linalg.norm(updated_particle)
            updated_particles.append(updated_particle)
            total_displacement += np.linalg.norm(max_displacement * displacement)
        particles = np.array(updated_particles)
        if total_displacement / len(particles) < convergence_threshold:
            return particles, True
    return particles, False


def visualize_point_clouds(initial_points, final_points):
    initial_pcd = o3d.geometry.PointCloud()
    initial_pcd.points = o3d.utility.Vector3dVector(initial_points)
    initial_pcd.paint_uniform_color([0, 0, 1])

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_points)
    final_pcd.paint_uniform_color([1, 0, 0])

    initial_pcd.translate([-1, 0, 0])
    final_pcd.translate([1, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Comparison", width=1200, height=800)

    vis.add_geometry(initial_pcd)
    vis.add_geometry(final_pcd)

    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 1, 0])
    view_control.set_front([0, 0, -1])
    view_control.set_zoom(0.5)

    vis.run()
    vis.destroy_window()


parser = argparse.ArgumentParser(
    description="Generate spherical blue noise or golden spiral distribution."
)
parser.add_argument('n', type=int, help='The number of points to generate.')
parser.add_argument(
    'distribution',
    type=str,
    choices=["spherical_blue_noise", "golden_spiral"],
    help='The type of distribution to generate.',
)
args = parser.parse_args()
n = args.n
distribution = args.distribution

initial_points = fibonacci_sphere(samples=2*n)
max_displacement = 0.1
iterations = 1000 if distribution == "spherical_blue_noise" else 0  # 0 iterations for golden spiral 
convergence_threshold = 1e-2

final_points, converged = advance_spherical_blue_noise(
    initial_points, max_displacement, iterations, convergence_threshold
)

if distribution == "spherical_blue_noise":
    if converged:
        print("The result converged.")
    else:
        print("The result did not converge within the specified number of iterations.")

# visualize_point_clouds(initial_points, final_points)

final_points = final_points[final_points[:, 2] > 0]
assert len(final_points) == n
np.save(f"data/{distribution}_{n}.npy", final_points)
print(f"Saved data/{distribution}_{n}.npy")