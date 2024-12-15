import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvDense2D(EnvBase):

    def __init__(self,
                 name='EnvDense2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):

        sphere_centers = np.array([[-0.43, 0.33], [0.33, 0.62]])
        sphere_radii = np.array([0.125, 0.125])
        sphere_velocities = np.array([[0.1, 0.2], [-0.1, 0.2]])

        obj_list = [
            MultiSphereField(
                sphere_centers,
                sphere_radii,
                velocities=sphere_velocities,
                tensor_args=tensor_args
            ),
            MultiBoxField(
                np.array(
                [[0.607781708240509, 0.19512386620044708], [0.5575312972068787, 0.5508843064308167],
                 [-0.3352295458316803, -0.6887519359588623], [-0.6572632193565369, 0.31827881932258606],
                 [-0.664594292640686, -0.016457155346870422], [0.8165988922119141, -0.19856023788452148],
                 [-0.8222246170043945, -0.6448580026626587], [-0.2855989933013916, -0.36841487884521484],
                 [-0.8946458101272583, 0.8962447643280029], [-0.23994405567646027, 0.6021060943603516],
                 [-0.006193588487803936, 0.8456171751022339], [0.305103600025177, -0.3661990463733673],
                 [-0.10704007744789124, 0.1318950206041336], [0.7156378626823425, -0.6923345923423767]
                 ]
                ),
                np.array(
                [[0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224]
                 ]
                )
                ,
                tensor_args=tensor_args
                )
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, 'dense2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=50
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    # Create environment with dynamic obstacles
    env = EnvDense2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS,
        dt=0.1  # 0.1 seconds per timestep
    )

    # Create a figure that updates
    plt.ion()  # Enable interactive mode
    fig, ax = create_fig_and_axes(env.dim)
    
    # Simulate for 50 timesteps
    for t in range(50):
        # Clear the axis
        ax.clear()
        
        # Render current state
        env.render(ax)
        ax.set_title(f'Time: {env.current_time:.1f}s')
        
        # Update display
        plt.draw()
        plt.pause(0.1)  # Pause to see the animation
        
        # Step environment
        env.step()
    
    plt.ioff()  # Disable interactive mode
    plt.show()

    # Optional: You can also visualize the SDF at any timestep
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)
    plt.show()

    # And the gradient of the SDF
    fig, ax = create_fig_and_axes(env.dim)
    env.render_grad_sdf(ax, fig)
    plt.show()
