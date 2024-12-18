import os
import pickle
import time
import numpy as np
import sys

import torch
import yaml
from matplotlib import pyplot as plt

from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))  # Go up 5 levels

print("Current directory:", current_dir)
print("Project root directory:", project_root)

# Add project root to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

print("Starting imports...")

# Debug: Try to find the actual module containing these functions
import deps.experiment_launcher.experiment_launcher as experiment_launcher
print("Contents of experiment_launcher:", dir(experiment_launcher))

try:
    # Import directly from decorators module
    from deps.experiment_launcher.experiment_launcher.decorators import single_experiment_yaml
    from deps.experiment_launcher.experiment_launcher.launcher import run_experiment
except ImportError as e:
    print("ImportError:", e)
    print("Check if the module path is correct and if there are circular dependencies.")
    raise

print("Imports successful.")

from deps.experiment_launcher.experiment_launcher.utils import fix_random_seed
from deps.motion_planning_baselines.mp_baselines.planners.gpmp2 import GPMP2
from deps.motion_planning_baselines.mp_baselines.planners.hybrid_planner import HybridPlanner
from deps.motion_planning_baselines.mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from deps.motion_planning_baselines.mp_baselines.planners.rrt_connect import RRTConnect



def get_obstacle_states_from_env(env):
    """Fallback method to get obstacle states from environment"""
    if hasattr(env, 'get_obstacle_states'):
        return env.get_obstacle_states()
    
    # Fallback: manually collect from obj_extra_list
    obstacle_states = {'positions': [], 'velocities': []}
    
    if hasattr(env, 'obj_extra_list') and env.obj_extra_list is not None:
        for obj in env.obj_extra_list:
            if hasattr(obj, 'fields'):
                for field in obj.fields:
                    if hasattr(field, 'centers'):  # For spheres and other primitives
                        obstacle_states['positions'].append(field.centers)
                        if hasattr(field, 'velocities'):
                            obstacle_states['velocities'].append(field.velocities)
    
    # Convert lists to tensors if any obstacles were found
    if len(obstacle_states['positions']) > 0:
        obstacle_states['positions'] = torch.cat(obstacle_states['positions'], dim=0)
        if len(obstacle_states['velocities']) > 0:
            obstacle_states['velocities'] = torch.cat(obstacle_states['velocities'], dim=0)
        return obstacle_states
    
    return {}


def generate_full_episodes_with_moving_obstacles(
    env_id,
    robot_id,
    results_dir,
    num_trajectories_per_context=1,
    threshold_start_goal_pos=1.0,
    obstacle_cutoff_margin=0.03,
    n_tries=1000,
    rrt_max_time=300,
    gpmp_opt_iters=500,
    n_support_points=64,
    duration=5.0,
    max_steps=100,  # max number of steps per episode to prevent infinite loops
    goal_reach_threshold=0.05,  # threshold to consider goal reached
    tensor_args=None,
    debug=False,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No suitable start/goal found.")

    # -------------------------------- Planner Setup ---------------------------------
    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state_pos=start_state_pos,
        goal_state_pos=goal_state_pos,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
    sample_based_planner = MultiSampleBasedPlanner(
        sample_based_planner_base,
        n_trajectories=num_trajectories_per_context,
        max_processes=-1,
        optimize_sequentially=True
    )

    # Optimization-based planner
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_trajectories_per_context,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    planner = HybridPlanner(
        sample_based_planner,
        opt_based_planner,
        tensor_args=tensor_args
    )

    # -------------------------------- Episode Simulation ---------------------------------
    # We'll simulate a full episode:
    # At each timestep:
    # 1. Get current state (agent pos, goal, obstacle positions/velocities)
    # 2. Plan full trajectory to goal
    # 3. Execute one step
    # 4. Update obstacles, store data
    # until goal is reached or max steps
    agent_pos = start_state_pos.clone()
    all_agent_positions = []
    all_obstacle_positions = []
    all_obstacle_velocities = []

    # initial states
    # Assuming environment or task has a method to get obstacle states
    # If not, you may need to query env.obj_extra_list or env.obj_fixed_list that hold dynamic objects
    initial_obstacle_state = get_obstacle_states_from_env(env)
    initial_obstacle_positions = initial_obstacle_state.get('positions', None)
    initial_obstacle_velocities = initial_obstacle_state.get('velocities', None)

    # store initial state
    initial_state_dict = {
        'start': start_state_pos.cpu().numpy(),
        'goal': goal_state_pos.cpu().numpy(),
        'obstacle_positions': initial_obstacle_positions.cpu().numpy() if initial_obstacle_positions is not None else None,
        'obstacle_velocities': initial_obstacle_velocities.cpu().numpy() if initial_obstacle_velocities is not None else None
    }

    # Lists to store the sequence of states
    # We'll store at each step:
    #   agent position
    #   obstacle positions
    #   (velocities stay the same if constant; if they change, store them too)
    for step_i in range(max_steps):
        # Update planner params with current start and goal
        planner.sample_based_planner.planner.start_state_pos = agent_pos
        planner.opt_based_planner.start_state = agent_pos
        planner.opt_based_planner.multi_goal_states = goal_state_pos.unsqueeze(0)

        # Plan trajectory
        trajs_iters = planner.optimize(debug=debug, print_times=False, return_iterations=True)
        trajs_last_iter = trajs_iters[-1]

        # Pick a trajectory (if multiple)
        chosen_traj = trajs_last_iter[0]  # assuming the first is good enough

        # Check if a solution was found
        # If no feasible trajectory found, break or handle failure
        if chosen_traj is None:
            print("No trajectory found at step:", step_i)
            break

        # Extract next step
        # The trajectory is shape: (time, dof)
        # We execute just the next configuration
        next_agent_pos = chosen_traj[1].clone()  # the first config is current state, second is next step
        all_agent_positions.append(agent_pos.cpu().numpy())

        # Store obstacle states at this step (before updating)
        current_obstacle_state = get_obstacle_states_from_env(env)
        if current_obstacle_state:
            all_obstacle_positions.append(current_obstacle_state['positions'].cpu().numpy())
            all_obstacle_velocities.append(current_obstacle_state['velocities'].cpu().numpy())

        # Update agent position
        agent_pos = next_agent_pos

        # Check if goal reached
        if torch.linalg.norm(agent_pos - goal_state_pos) < goal_reach_threshold:
            print("Goal reached at step:", step_i)
            all_agent_positions.append(agent_pos.cpu().numpy())
            # Store last obstacle state too
            current_obstacle_state = get_obstacle_states_from_env(env)
            if current_obstacle_state:
                all_obstacle_positions.append(current_obstacle_state['positions'].cpu().numpy())
                all_obstacle_velocities.append(current_obstacle_state['velocities'].cpu().numpy())
            break

        # Move environment forward (obstacles move)
        env.step()

    # Convert all stored data into arrays
    all_agent_positions = np.array(all_agent_positions)
    all_obstacle_positions = np.array(all_obstacle_positions) if len(all_obstacle_positions) > 0 else None
    all_obstacle_velocities = np.array(all_obstacle_velocities) if len(all_obstacle_velocities) > 0 else None

    # Save the collected episode data
    episode_data = {
        'initial_state': initial_state_dict,
        'agent_positions': all_agent_positions,
        'obstacle_positions_sequence': all_obstacle_positions,
        'obstacle_velocities_sequence': all_obstacle_velocities
    }

    with open(os.path.join(results_dir, f'episode_data.pickle'), 'wb') as handle:
        pickle.dump(episode_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional: visualize the final trajectory
    planner_visualizer = PlanningVisualizer(task=task)
    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=torch.tensor(all_agent_positions, device=tensor_args['device']).unsqueeze(0),
        pos_start_state=all_agent_positions[0],
        pos_goal_state=goal_state_pos.cpu().numpy(),
        vel_start_state=np.zeros_like(all_agent_positions[0]),
        vel_goal_state=np.zeros_like(goal_state_pos.cpu().numpy()),
    )
    fig.savefig(os.path.join(results_dir, f'episode_trajectory.png'), dpi=300)
    plt.close(fig)

    return True


@single_experiment_yaml
def experiment(
    env_id: str = 'EnvDense2D',
    robot_id: str = 'RobotPointMass',
    n_support_points: int = 64,
    duration: float = 5.0,  # seconds
    threshold_start_goal_pos: float = 1.0,
    obstacle_cutoff_margin: float = 0.05,
    num_trajectories: int = 1,
    device: str = 'cuda',
    debug: bool = True,
    seed: int = int(time.time()),
    results_dir: str = f"data",
    **kwargs
):
    if debug:
        fix_random_seed(seed)

    print(f'\n\n-------------------- Generating full episode data --------------------')
    print(f'Seed:  {seed}')
    print(f'Env:   {env_id}')
    print(f'Robot: {robot_id}')
    print(f'num_trajectories: {num_trajectories}')

    tensor_args = {'device': device, 'dtype': torch.float32}

    metadata = {
        'env_id': env_id,
        'robot_id': robot_id,
        'num_trajectories': num_trajectories,
        'episode_type': 'dynamic_obstacles_full_episode'
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)

    success = generate_full_episodes_with_moving_obstacles(
        env_id,
        robot_id,
        results_dir=results_dir,
        num_trajectories_per_context=num_trajectories,
        threshold_start_goal_pos=threshold_start_goal_pos,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        n_support_points=n_support_points,
        duration=duration,
        tensor_args=tensor_args,
        debug=debug,
    )

    if success:
        print("Episode generation completed successfully.")
    else:
        print("Episode generation failed or incomplete.")


if __name__ == '__main__':
    run_experiment(experiment)
