#!/usr/bin/env python3
#Standard Libraries
import numpy as np
from sklearn.neighbors import KDTree
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
# def disk(center, radius, shape=None):
#     """Generate disk mask for given center and radius."""
#     rr, cc = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
#     mask = rr**2 + cc**2 <= radius**2
#     return np.where(mask)
from scipy.linalg import block_diag

COLORS = dict(
    w = (255, 255, 255),
    k = (0, 0, 0),
    g = (0, 255, 0),
    r = (255, 0, 0),
    b = (0, 0, 255)
)

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        self.child_trajectories = dict()
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.26 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m
        self.optimal_goal_id = -1
        self.goal_ids = []

        #Trajectory Simulation Parameters
        # willowworld
        self.timestep = 3.0 #s
        self.num_substeps = 20
        # myhal
        # self.timestep = 3.0
        # self.num_substeps = 20

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.vis_radius = 3
        # Willowworld
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (800, 800), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        # Myhal
        # print(self.occupancy_map.shape)
        # real_map_size_pixels = [self.occupancy_map.shape[1], self.occupancy_map.shape[0]]
        # self.window = pygame_utils.PygameWindow(
        #     "Path Planner", (159*10, 49*10), real_map_size_pixels, self.map_settings_dict, self.goal_point, self.stopping_dist)
        print(f"Map bounds: {self.bounds}")

        return

    #Functions required for RRT
    def sample_map_space(self):
        """
        Sample a random (x, y) point within the free space of the map.
        4% of the time, bias the sampling towards the goal or near the goal (3x-5x stopping_dist away).

        Returns:
            numpy.ndarray: A (2,) array representing the sampled [x, y] coordinates.
        """
        rand_num = np.random.rand()  # Generate a random number between 0 and 1

        if rand_num < 0.04:  # 4% probability of goal bias
            if np.random.rand() < 0.5:  # 50% chance of choosing goal
                return self.goal_point.flatten()
            else:  # 50% chance of choosing near goal
                theta = np.random.uniform(0, 2 * np.pi)  # Random direction
                offset_dist = np.random.uniform(3, 5) * self.stopping_dist  # Fixed range (3x-5x stopping_dist)
                offset = offset_dist * np.array([np.cos(theta), np.sin(theta)])
                near_goal = self.goal_point.flatten() + offset

                # Ensure near-goal sample is within bounds
                near_goal[0] = np.clip(near_goal[0], self.bounds[0, 0], self.bounds[0, 1])
                near_goal[1] = np.clip(near_goal[1], self.bounds[1, 0], self.bounds[1, 1])

                return near_goal

        else:  # 96% probability of random sampling
            while True:
                # Sample x and y within map bounds
                # myhal
                # x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
                # y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
                # willow
                x = np.random.uniform(0, 43)
                y = np.random.uniform(-45, 3)

                # Convert to occupancy grid coordinates
                cell = self.point_to_cell(np.array([[x, y]]))[0]

                # Check if point is in free space (1 = free, 0 = occupied)
                if self.occupancy_map[cell[0], cell[1]] == 1:
                    return np.array([x, y])
    
    def check_if_duplicate(self, point):
        """
        Check if a point is a duplicate of an existing node.

        Args:
            point (numpy.ndarray): A (3,) array representing [x, y, theta].

        Returns:
            bool: True if the point is a duplicate, False otherwise.
        """
        if len(self.nodes) == 0:
            return False  # No nodes yet, no duplicates possible

        # Convert node list to NumPy array for KDTree
        node_poses = np.array([node.point.flatten() for node in self.nodes])  # Shape (N, 3)

        # Build KDTree for fast nearest-neighbor search
        tree = KDTree(node_poses)

        # Ensure point is 2D for KDTree query
        point = np.atleast_2d(point)  # Converts (3,) -> (1,3) if needed

        # Query the nearest neighbor
        dist, _ = tree.query(point, k=1)  # Get nearest node

        return dist < 1e-3  # Small tolerance to avoid floating-point errors
    
    def closest_node(self, point):
        """
        Find the closest node in the tree to a given point using KDTree.

        Args:
            point (numpy.ndarray): A (2,) array representing [x, y].

        Returns:
            int: Index of the closest node in `self.nodes`.
        """
        if len(self.nodes) == 0:
            raise ValueError("No nodes in the tree!")

        # Convert node list to NumPy array for KDTree
        node_poses = np.array([node.point[:2].flatten() for node in self.nodes])  # Shape (N, 2)

        # Build KDTree on-the-fly
        tree = KDTree(node_poses)

        # Ensure point is 2D for KDTree query
        point = np.atleast_2d(point)  # Converts (2,) -> (1,2) if needed

        # Query KDTree for nearest neighbor
        _, idx = tree.query(point, k=1)
        return int(idx[0, 0])

    def simulate_trajectory(self, node_i, point_s):
        """
        Simulates the non-holonomic motion of the robot from a given node towards a target point.

        Args:
            node_i (numpy.ndarray): A (3,) array representing [x, y, theta] of the current robot state.
            point_s (numpy.ndarray): A (2,) array representing the target point.

        Returns:
            numpy.ndarray: (3, num_substeps) trajectory from `node_i` towards `point_s`.
            float: Best linear velocity.
            float: Best rotational velocity.
        """
        # Get best control velocities
        # best_vel, best_rot_vel, valid_trajectories, last_valid_indices = self.robot_controller(node_i, point_s)

        # # Generate the best trajectory using the optimal (best_vel, best_rot_vel)
        # best_trajectory = self.trajectory_rollout(node_i, np.array([best_vel]), np.array([best_rot_vel]))[0]  # (3, num_substeps)

        # return best_trajectory, best_vel, best_rot_vel
        best_trajectory = self.robot_controller(node_i, point_s)
        return best_trajectory
    
    def robot_controller(self, node_i, point_s):
        """
        Generate velocity options, simulate trajectories, and check for collisions.

        Args:
            node_i (numpy.ndarray): A (3,) array representing [x, y, theta] of the current robot state.
            point_s (numpy.ndarray): A (2,) array representing the target point.

        Returns:
            best_vel (float): Best linear velocity.
            best_rot_vel (float): Best angular velocity.
            valid_trajectories (numpy.ndarray): (MN, 3, num_substeps) array of valid trajectories.
            last_valid_indices (numpy.ndarray): (MN,) array of last valid time step before collision.
        """
        # Define velocity ranges
        lin_vels = np.linspace(-self.vel_max, self.vel_max, num=10)  # M options
        rot_vels = np.linspace(-self.rot_vel_max, self.rot_vel_max, num=10)  # N options

        # Create velocity meshgrid
        lin_grid, rot_grid = np.meshgrid(lin_vels, rot_vels)
        lin_grid = lin_grid.ravel()  # (MN,)
        rot_grid = rot_grid.ravel()  # (MN,)
        MN = lin_grid.shape[0]  # Total number of velocity pairs

        # Compute all trajectories in parallel
        trajectories = self.trajectory_rollout(node_i, lin_grid, rot_grid)  # Shape: (MN, 3, num_substeps)

        # Perform collision checking
        # BEFORE START
        # valid_trajectories, last_valid_indices = self.check_collisions(trajectories)

        # # Extract last valid position for each trajectory
        # final_x = np.take_along_axis(valid_trajectories[:, 0, :], last_valid_indices[:, None], axis=1).squeeze()
        # final_y = np.take_along_axis(valid_trajectories[:, 1, :], last_valid_indices[:, None], axis=1).squeeze()

        # # Compute final position error (Euclidean distance)
        # final_error = np.sqrt((final_x - point_s[0])**2 + (final_y - point_s[1])**2)
        # BEFORE END
        
        # AFTER START
        valid_trajectories, last_valid_indices, last_valid_poses = self.check_collisions(trajectories)
        final_x = last_valid_poses[:, 0]  # Last valid x-coordinates
        final_y = last_valid_poses[:, 1]  # Last valid y-coordinates
        final_error = np.sqrt((final_x - point_s[0])**2 + (final_y - point_s[1])**2)
        # AFTER END

        # Select the trajectory with the smallest final position error
        best_idx = np.nanargmin(final_error)  # Pick trajectory that minimizes error
        best_trajectory = valid_trajectories[best_idx]
        last_valid_idx = last_valid_indices[best_idx]
        best_trajectory = best_trajectory[:, :last_valid_idx + 1]
        return best_trajectory

    def trajectory_rollout(self, initial_pose, lin_vels, rot_vels):
        """
        Compute trajectories for multiple velocity combinations using vectorization.

        Args:
            initial_pose (numpy.ndarray): A (3,) array representing [x_0, y_0, theta_0].
            lin_vels (numpy.ndarray): (MN,) array of linear velocities.
            rot_vels (numpy.ndarray): (MN,) array of rotational velocities.

        Returns:
            numpy.ndarray: (MN, 3, num_substeps) matrix of trajectories.
        """
        MN = lin_vels.shape[0]
        dt = self.timestep / self.num_substeps  # Time step per substep
        t = np.linspace(dt, self.timestep, self.num_substeps)  # Time steps (1D array)

        x_0, y_0, theta_0 = initial_pose  # Extract initial pose (single instance)

        # Expand for batch processing
        theta_0 = np.repeat(theta_0, MN)  # (MN,)
        x_0 = np.repeat(x_0, MN)
        y_0 = np.repeat(y_0, MN)
        t = np.tile(t, (MN, 1))  # Shape: (MN, num_substeps)

        # Compute trajectories in vectorized form
        moving = np.abs(rot_vels) > 1e-6  # Mask for rotational motion

        theta_t = theta_0[:, None] + rot_vels[:, None] * t  # (MN, num_substeps)

        # Rotational motion equations
        x_t = np.where(
            moving[:, None],
            x_0[:, None] + (lin_vels[:, None] / rot_vels[:, None]) * (np.sin(theta_t) - np.sin(theta_0[:, None])),
            x_0[:, None] + lin_vels[:, None] * t * np.cos(theta_0[:, None])
        )

        y_t = np.where(
            moving[:, None],
            y_0[:, None] - (lin_vels[:, None] / rot_vels[:, None]) * (np.cos(theta_t) - np.cos(theta_0[:, None])),
            y_0[:, None] + lin_vels[:, None] * t * np.sin(theta_0[:, None])
        )

        return np.stack((x_t, y_t, theta_t), axis=1)  # Shape: (MN, 3, num_substeps)

    def is_collision(self, point): # only used for testing 
        """
        Check if a given (x, y) point is colliding with an obstacle or is out of bounds.

        Args:
            point (numpy.ndarray): Shape (2,), representing (x, y) coordinates.

        Returns:
            bool: True if the point is in collision, False otherwise.
        """
        # Convert (x, y) to occupancy grid indices
        cell = self.point_to_cell(np.array([point]))[0]  # Convert to int cell index

        # Ensure within bounds
        if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.occupancy_map.shape[0] or cell[1] >= self.occupancy_map.shape[1]:
            return True  # Out of bounds means collision

        # Get robot's footprint at this position
        footprint = self.points_to_robot_circle(np.array([point]))[0]  # Shape: (num_circ_px, 2)

        # Check if any part of the footprint is in an obstacle
        for px in footprint:
            row, col = px
            if self.occupancy_map[row, col] == 0:  # 0 indicates an obstacle
                return True  # Collision detected

        return False  # No collision

    def check_collisions(self, trajectories):
        """
        Perform fast collision checking for multiple trajectories.

        Args:
            trajectories (numpy.ndarray): Shape (MN, 3, num_substeps), where each trajectory contains
                                        (x, y, theta) for every timestep.

        Returns:
            valid_trajectories (numpy.ndarray): Same shape as `trajectories` but with NaNs after first collision.
            last_valid_indices (numpy.ndarray): (MN,) array with the last valid timestep before collision.
        """
        MN, _, num_substeps = trajectories.shape
        valid_trajectories = np.copy(trajectories)  # Copy original to modify
        last_valid_indices = np.full(MN, num_substeps - 1, dtype=int)  # Default: last valid step is last step

        # Iterate through all trajectories
        for i in range(MN):
            first_collision_step = num_substeps  # Set to out of range initially

            # Iterate through each step in the trajectory
            for j in range(num_substeps):
                x, y, _ = trajectories[i, :, j]  # Extract (x, y, theta) at step j

                # Convert (x, y) to occupancy grid indices
                cell = self.point_to_cell(np.array([[x, y]]))[0]  # Convert to int cell index

                # Ensure within bounds
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.occupancy_map.shape[0] or cell[1] >= self.occupancy_map.shape[1]:
                    first_collision_step = j
                    break  # If out of bounds, it's a collision

                # Get robot's footprint at this step
                footprint = self.points_to_robot_circle(np.array([[x, y]]))[0]  # Shape: (num_circ_px, 2)

                # Check if any part of the footprint is in an obstacle (0 in occupancy map)
                for px in footprint:
                    row, col = px
                    if self.occupancy_map[row, col] == 0:  # Obstacle hit
                        first_collision_step = j
                        break

                if first_collision_step < num_substeps:  # If collision found, stop checking further
                    break

            # If collision happened, invalidate the rest of the trajectory
            if first_collision_step < num_substeps:
                valid_trajectories[i, :, first_collision_step:] = np.nan  # Set rest to NaN
                last_valid_indices[i] = first_collision_step - 1  # Set last valid step
        
        last_valid_poses = valid_trajectories[np.arange(MN), :, last_valid_indices]  # Shape: (MN, 3)
        return valid_trajectories, last_valid_indices, last_valid_poses

    def point_to_cell(self, points):
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # points is an N x 2 matrix of points of interest

        resolution = self.map_settings_dict["resolution"]
        origin = self.map_settings_dict["origin"]

        cols = ((points[:, 0] - origin[0]) / resolution).astype(int)
        rows = ((self.bounds[1, 1] - points[:, 1]) / resolution).astype(int)  # Flip Y-axis

        x = np.column_stack((rows, cols)) # (N, 2)
        return x

    def points_to_robot_circle(self, points):
        """
        Convert a series of [x, y] points to robot map footprints for collision detection.

        Args:
            points (numpy.ndarray): An (N, 2) array of points.

        Returns:
            numpy.ndarray: An (N, num_circ_px, 2) array of robot footprints.
        """
        resolution = self.map_settings_dict["resolution"]
        occupancy_shape = self.occupancy_map.shape
        radius_in_cells = int(self.robot_radius / resolution)

        # Precompute the disk footprint at (0, 0)
        rr, cc = disk((0, 0), radius_in_cells)
        offsets = np.column_stack((rr, cc))  # Shape: (num_circ_px, 2)

        # Convert all points to cell coordinates (Now returns (N, 2))
        cell_coords = self.point_to_cell(points)  # No need for apply_along_axis

        # Apply offsets to all points using broadcasting
        footprints = cell_coords[:, None, :] + offsets[None, :, :]  # Shape: (N, num_circ_px, 2)

        # Bounds checking
        footprints[..., 0] = np.clip(footprints[..., 0], 0, occupancy_shape[0] - 1)
        footprints[..., 1] = np.clip(footprints[..., 1], 0, occupancy_shape[1] - 1)

        return footprints  # Shape: (N, num_circ_px, 2)
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, start_node, target_pos):
        """
        Computes a feasible non-holonomic trajectory to connect a node to a given target position.

        Args:
            start_node (numpy.ndarray): A (3, 1) array representing [x, y, theta] of the starting node.
            target_pos (numpy.ndarray): A (2,) array representing the goal (x, y) position.

        Returns:
            numpy.ndarray: (3, num_substeps) trajectory connecting `start_node` to `target_pos`.
        """
        # Extract initial position and orientation
        start_state = start_node.point.flatten()  # Convert to (3,)
        x_start, y_start, theta_start = start_state

        # Convert goal to the robot’s local frame
        world_to_robot = np.array([
            [np.cos(theta_start), -np.sin(theta_start), x_start],
            [np.sin(theta_start), np.cos(theta_start), y_start],
            [0, 0, 1]
        ])
        robot_to_world = np.linalg.inv(world_to_robot)

        target_homogeneous = np.array([target_pos[0], target_pos[1], 1.0])
        target_local = robot_to_world @ target_homogeneous  # Transform into robot's frame

        x_local_goal, y_local_goal = target_local[:2]  # Extract (x, y) in local frame

        # If the target is directly ahead, drive straight
        if np.isclose(y_local_goal, 0.0):
            final_theta = theta_start
            trajectory = np.linspace(start_state, np.array([target_pos[0], target_pos[1], final_theta]), self.num_substeps).T
        else:
            # Compute turning radius for circular path
            turn_radius = (x_local_goal**2 + y_local_goal**2) / (2 * y_local_goal)
            circle_center_local = np.array([0.0, turn_radius, 1])
            circle_center_world = (world_to_robot @ circle_center_local)[:2]

            # Determine velocity commands
            ang_speed = self.rot_vel_max
            lin_speed = turn_radius * ang_speed
            if lin_speed > self.vel_max:
                lin_speed = self.vel_max
                ang_speed = lin_speed / turn_radius

            # Compute required turning duration
            theta_target = np.arctan2(
                (target_pos[0] - x_start) / turn_radius + np.sin(theta_start),
                -(target_pos[1] - y_start) / turn_radius + np.cos(theta_start),
            )
            theta_delta = (theta_target - theta_start + np.pi) % (2 * np.pi) - np.pi
            time_needed = np.abs(theta_delta) / ang_speed
            if theta_delta < 0:
                ang_speed *= -1
                lin_speed *= -1

            # Generate trajectory
            time_steps = np.linspace(0, time_needed, self.num_substeps)
            x_path = np.where(
                np.isclose(ang_speed, 0),
                x_start + lin_speed * time_steps * np.cos(theta_start),
                x_start + (lin_speed / ang_speed) * (np.sin(ang_speed * time_steps + theta_start) - np.sin(theta_start))
            )
            y_path = np.where(
                np.isclose(ang_speed, 0),
                y_start + lin_speed * time_steps * np.sin(theta_start),
                y_start - (lin_speed / ang_speed) * (np.cos(ang_speed * time_steps + theta_start) - np.cos(theta_start))
            )
            theta_path = theta_start + ang_speed * time_steps

            # Normalize angles between -π and π
            theta_path = (theta_path + np.pi) % (2 * np.pi) - np.pi

            trajectory = np.stack((x_path, y_path, theta_path), axis=0)  # Shape (3, num_substeps)

        # Collision check and truncation
        valid_paths, valid_indices, valid_positions = self.check_collisions(trajectory[None, :, :])
        last_safe_idx = valid_indices[0]  # Get last valid step
        last_safe_pose = valid_positions[0]  # Last valid pose before collision

        return valid_paths[0, :, :last_safe_idx + 1].T, last_safe_idx, last_safe_pose

    def cost_to_come(self, traj_points):
        """
        Computes the cost of reaching a node along a given trajectory.

        Args:
            traj_points (numpy.ndarray): Shape (N, 3), representing (x, y, theta) states.

        Returns:
            float: The total path cost based on translational and rotational effort.
        """
        if traj_points.shape[0] < 2:
            return 0  # If trajectory is too short, cost is zero
        # Compute distance traveled in (x, y) space
        step_distances = np.linalg.norm(traj_points[1:, :2] - traj_points[:-1, :2], axis=1)
        translation_cost = step_distances.sum()

        # Compute orientation changes
        theta_diff = np.abs(traj_points[1:, 2] - traj_points[:-1, 2])
        angular_error = np.minimum(2 * np.pi - theta_diff, theta_diff)  # Normalize rotation difference
        rotation_cost = np.abs(angular_error).sum() * self.robot_radius

        return translation_cost + 0.101 * rotation_cost  # Weighted sum of translation & rotation costs

    def update_children(self, parent_id):
        """
        Updates the cost of all child nodes recursively after a rewire.

        Args:
            parent_id (int): Index of the parent node whose cost has changed.
        """
        parent_node = self.nodes[parent_id]  # Retrieve parent node details

        # Iterate through all child nodes
        for child_idx in parent_node.children_ids:
            # Get trajectory from parent to child
            path_to_child = parent_node.child_trajectories[child_idx]  # Different name from "pose_traj_to_children"
            
            # Compute additional cost for this edge
            edge_cost = self.cost_to_come(path_to_child)

            # Update child's cost
            self.nodes[child_idx].cost = parent_node.cost + edge_cost

            # Recursively update deeper children
            self.update_children(child_idx)

    def rrt_planning(self):
        """
        Run the RRT algorithm to find a path from start to goal.

        Returns:
            list: List of nodes representing the path from start to goal.
        """
        while True:
            # Sample a new point
            point = self.sample_map_space()
            # self.window.add_point(point.copy(), radius=self.vis_radius, color=COLORS['g'])  # Blue nodes

            # Find the closest existing node
            closest_idx = self.closest_node(point)
            closest_node = self.nodes[closest_idx]

            # Generate a trajectory towards the sampled point
            new_trajectory = self.simulate_trajectory(closest_node.point.flatten(), point)

            # Get final position and orientation from trajectory
            new_point = new_trajectory[:2, -1]  # Extract final (x, y)
            new_theta = new_trajectory[2, -1]   # Extract final theta
            if self.is_collision(new_point):
                print("Collision detected!")

            # Check for duplicates
            if self.check_if_duplicate(np.array([new_point[0], new_point[1], new_theta])):
                print(f"Duplicate rejected")
                continue

            # Create new node with updated theta
            new_node = Node(
                np.array([[new_point[0]], [new_point[1]], [new_theta]]),  # Properly update theta
                parent_id=closest_idx,
                cost=0
            )

            # Add the new node to the tree
            self.nodes.append(new_node)
            new_node_idx = len(self.nodes) - 1

            # Update parent node to store the new child
            closest_node.children_ids.append(len(self.nodes) - 1)

            self.window.add_point(new_point.copy(), radius=self.vis_radius, color=COLORS['b'])  # Blue nodes
            self.window.add_line(closest_node.point[:2].flatten(), new_point.copy(), width=1, color=COLORS['g'])  # Green for tree edges

            # Check if goal is reached
            if np.linalg.norm(new_point - self.goal_point.flatten()) < self.stopping_dist:
                print("Goal reached!")
                best_path = self.recover_path(new_node_idx)
                    
                for i in range(len(best_path) - 1):
                    self.window.add_line(best_path[i][:2].flatten(), best_path[i + 1][:2].flatten(), width=4, color=COLORS['r'])  # Red for final path

                node_path_metric = np.hstack(best_path)
                np.save("shortest_path_myhal.npy", node_path_metric)
                input()
                return best_path
    
    def rrt_star_planning(self, max_iterations=40000):
        """
        Executes the RRT* algorithm to find an optimized path to the goal.

        Args:
            max_iterations (int): The maximum number of iterations before stopping.

        Returns:
            list: The final set of nodes representing the RRT* tree.
        """
        iteration_count = 0

        while 0<1:
            iteration_count += 1
            # print(f'iteration {iteration_count}')

            # Generate a random sample in free space
            sampled_point = self.sample_map_space()
            # print(f'sampled at {sampled_point}')
            # time.sleep(1)
            # self.window.add_point(sampled_point.copy(), radius=3, color=[160,32,240])  # Green sample points

            # Find the nearest existing node in the tree
            nearest_node_idx = self.closest_node(sampled_point)
            nearest_node = self.nodes[nearest_node_idx]

            # Simulate motion from nearest node towards sampled point
            trajectory_to_new = self.simulate_trajectory(nearest_node.point.flatten(), sampled_point)
            # Extract the final position in the trajectory
            proposed_new_position = trajectory_to_new[:, -1]

            # Skip if this position is too close to an existing node
            if self.check_if_duplicate(proposed_new_position):
                continue

            # Compute cost-to-come for the new node
            new_node_cost = nearest_node.cost + self.cost_to_come(trajectory_to_new.T)

            # Create the new node and attach it to the tree
            new_node = Node(
                point=np.array([[proposed_new_position[0]], [proposed_new_position[1]], [proposed_new_position[2]]]),
                parent_id=nearest_node_idx,
                cost=new_node_cost
            )
            self.nodes.append(new_node)
            new_node_idx = len(self.nodes) - 1

            # Update parent-child relationships
            nearest_node.children_ids.append(new_node_idx)
            nearest_node.child_trajectories[new_node_idx] = trajectory_to_new.T
            
            self.window.add_point(proposed_new_position[:2].copy(), radius=self.vis_radius, color=COLORS['b'])  # Blue for new nodes
            self.window.add_line(nearest_node.point[:2].flatten(), proposed_new_position[:2].copy(), width=1, color=COLORS['g'])  # Green for tree edges

            # Search for better parent nodes within a radius
            search_tree = KDTree(np.array([n.point[:2].flatten() for n in self.nodes]))
            nearby_nodes = search_tree.query_radius(np.atleast_2d(proposed_new_position[:2]), self.ball_radius())[0]

            # Remove the newly added node from search results
            nearby_nodes = [idx for idx in nearby_nodes if idx != new_node_idx]

            best_parent_idx = None
            best_path_cost = new_node_cost
            best_trajectory = None

            for potential_parent_idx in nearby_nodes:
                potential_parent = self.nodes[potential_parent_idx]

                # Simulate connection from potential parent to new node
                candidate_trajectory, last_safe_idx, candidate_end_pose = self.connect_node_to_point(potential_parent, proposed_new_position[:2])

                # Ensure full trajectory is valid (i.e., no early termination due to collision)
                if last_safe_idx != self.num_substeps-1:
                    continue

                candidate_cost = potential_parent.cost + self.cost_to_come(candidate_trajectory)

                # If this new path is better, update best known path
                if candidate_cost < best_path_cost:
                    best_parent_idx = potential_parent_idx
                    best_path_cost = candidate_cost
                    best_trajectory = candidate_trajectory

            # If a better parent was found, rewire the new node
            if best_parent_idx is not None:
                # Remove old parent reference
                old_parent_idx = nearest_node_idx
                self.nodes[old_parent_idx].children_ids.remove(new_node_idx)
                del self.nodes[old_parent_idx].child_trajectories[new_node_idx]

                # Attach to the best new parent
                new_node.parent_id = best_parent_idx
                new_node.cost = best_path_cost
                self.nodes[best_parent_idx].children_ids.append(new_node_idx)
                self.nodes[best_parent_idx].child_trajectories[new_node_idx] = best_trajectory  # Update trajectory
                
                # Visualization: Remove old edge, add new one
                self.window.remove_line(self.nodes[old_parent_idx].point[:2].flatten(), proposed_new_position[:2].copy())
                self.window.add_line(self.nodes[best_parent_idx].point[:2].flatten(), proposed_new_position[:2].copy(), width=1, color=COLORS['g'])

            # Try rewiring other nodes in the vicinity to use the new node as their parent
            for nearby_idx in nearby_nodes:
                nearby_node = self.nodes[nearby_idx]

                # Generate a new trajectory from new_node to nearby_node
                potential_trajectory, p_last_index, potential_end_pose = self.connect_node_to_point(new_node, nearby_node.point[:2].flatten())

                if p_last_index != self.num_substeps-1:
                    continue

                transition_cost = self.cost_to_come(potential_trajectory)
                potential_new_cost = new_node.cost + transition_cost

                # If the new path reduces cost-to-come, rewire the node
                if potential_new_cost < nearby_node.cost:
                    old_parent_idx = nearby_node.parent_id
                    self.nodes[old_parent_idx].children_ids.remove(nearby_idx)
                    del self.nodes[old_parent_idx].child_trajectories[nearby_idx]

                    # Rewire to the new node
                    nearby_node.parent_id = new_node_idx
                    nearby_node.cost = potential_new_cost
                    new_node.children_ids.append(nearby_idx)
                    new_node.child_trajectories[nearby_idx] = potential_trajectory  # Store new trajectory

                    # Recursively update children costs
                    self.update_children(nearby_idx)
                    
                    # Visualization: Remove old edge, add new edge
                    self.window.remove_line(self.nodes[old_parent_idx].point[:2].flatten(), nearby_node.point[:2].copy())
                    self.window.add_line(new_node.point[:2].flatten(), nearby_node.point[:2].copy(), width=1, color=COLORS['g'])

            # Check if we have reached the goal
            if np.linalg.norm(proposed_new_position[:2] - self.goal_point.flatten()) <= self.stopping_dist:
                self.goal_ids.append(new_node_idx)
                if self.optimal_goal_id == -1:
                    print("Goal Reached for first time!")
                    self.optimal_goal_id = new_node_idx
                    # Recover path and highlight it
                    best_path = self.recover_path(new_node_idx)

                    for i in range(len(best_path) - 1):
                        self.window.add_line(best_path[i][:2].flatten(), best_path[i + 1][:2].flatten(), width=4, color=COLORS['r'])  # Red for final path

                    node_path_metric = np.hstack(self.recover_path(new_node_idx))
                    np.save("shortest_path_myhal.npy", node_path_metric)
            
            for node_id in self.goal_ids:
                if self.nodes[node_id].cost < self.nodes[self.optimal_goal_id].cost:
                    print("BETTER SOLUTION FOUND")
                    self.optimal_goal_id = node_id
                    best_path = self.recover_path(node_id)
                    
                    for i in range(len(best_path) - 1):
                        self.window.add_line(best_path[i][:2].flatten(), best_path[i + 1][:2].flatten(), width=4, color=COLORS['r'])  # Red for final path

                    node_path_metric = np.hstack(self.recover_path(node_id))
                    np.save("shortest_path_myhal.npy", node_path_metric)
                    # for i in range(len(best_path) - 1):
                    #     self.window.add_line(best_path[i][:2].flatten(), best_path[i + 1][:2].flatten(), width=2, color=COLORS['r'])  # Red for final path


        print("Max iterations reached. No solution found.")
        return self.nodes

    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def test_points2robotcircles(map_instance, points):
    """
    Visualize the occupancy grid with robot footprints.

    Args:
        map_instance: Instance of the map class with points_to_robot_circle function.
        points (numpy.ndarray): An (N, 2) array of robot positions in meters.
    
    Displays:
        A grid with 1s representing robot-occupied cells and 0s for empty space.
    """
    # Get occupancy grid size
    grid_shape = map_instance.occupancy_map.shape

    # Initialize empty grid
    grid = np.zeros(grid_shape, dtype=int)

    # Get robot footprints
    footprints = map_instance.points_to_robot_circle(points)  # Shape: (N, num_circ_px, 2)

    # Set robot-occupied cells to 1
    for footprint in footprints:
        for r, c in footprint:
            grid[r, c] = 1  # Mark footprint in the grid

    # Print grid
    print("\nOccupancy Grid:")
    for row in grid:
        print("".join(str(cell) for cell in row))

def test_trajectoryrollout(map_instance):
    """
    Test trajectory rollout
    """
    trajs = map_instance.trajectory_rollout(initial_pose=(0,0,0), lin_vels=np.array([0,1,2]), rot_vels=np.array([0,0,1]))
    print(trajs)

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42], [-44]]) #m # willowworld
    stopping_dist = 0.5
    # goal_point = np.array([[7], [0]]) #m #myhal
    # stopping_dist = 0.1 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path_willow.npy", node_path_metric)


if __name__ == '__main__':
    main()
