import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


class ArtificialPotentialField:


    def __init__(self, map_size=(100, 100),
                 attractive_gain=1.0, repulsive_gain=10.0,
                 obstacle_influence=10.0):
        self.map_size = map_size
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.obstacle_influence = obstacle_influence

        self.attractive_field = np.zeros(map_size)
        self.repulsive_field = np.zeros(map_size)
        self.total_field = np.zeros(map_size)

    def set_goal(self, goal_position):
        """设置目标点，计算引力场"""
        self.goal = np.array(goal_position)

        # 创建坐标网格
        x = np.arange(self.map_size[0])
        y = np.arange(self.map_size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # 计算到目标的距离
        distance = np.sqrt((X - goal_position[0]) ** 2 +
                           (Y - goal_position[1]) ** 2)

        # 引力场（与距离的平方成正比）
        self.attractive_field = 0.5 * self.attractive_gain * distance ** 2

    def set_obstacles(self, obstacle_map):
        """设置障碍物，计算斥力场"""
        self.obstacle_map = obstacle_map

        # 计算到最近障碍物的距离
        distance_to_obstacle = distance_transform_edt(1 - obstacle_map)

        # 斥力场（在影响范围内）
        mask = distance_to_obstacle <= self.obstacle_influence
        self.repulsive_field = np.zeros_like(distance_to_obstacle)

        # 计算斥力
        with np.errstate(divide='ignore'):
            inv_distance = 1.0 / (distance_to_obstacle + 1e-6)
            repulsive_force = 0.5 * self.repulsive_gain * \
                              (1.0 / distance_to_obstacle - 1.0 / self.obstacle_influence) ** 2
            self.repulsive_field[mask] = repulsive_force[mask]

        # 在障碍物位置设置最大斥力
        self.repulsive_field[obstacle_map > 0] = 1000

    def compute_total_field(self):
        """计算合势场"""
        self.total_field = self.attractive_field + self.repulsive_field
        return self.total_field

    def plan_path(self, start_position, max_iterations=1000,
                  step_size=1.0, goal_threshold=2.0):
        """规划路径"""
        current_pos = np.array(start_position)
        path = [current_pos.copy()]

        for i in range(max_iterations):
            # 检查是否到达目标
            if np.linalg.norm(current_pos - self.goal) < goal_threshold:
                print(f"到达目标！迭代次数: {i}")
                break

            # 计算当前位置的梯度
            gradient = self.compute_gradient(current_pos)

            # 沿梯度下降方向移动
            if np.linalg.norm(gradient) > 0:
                direction = -gradient / np.linalg.norm(gradient)
                current_pos = current_pos + direction * step_size

            # 边界检查
            current_pos[0] = np.clip(current_pos[0], 0, self.map_size[0] - 1)
            current_pos[1] = np.clip(current_pos[1], 0, self.map_size[1] - 1)

            path.append(current_pos.copy())

        return np.array(path)

    def compute_gradient(self, position):
        """计算势场的梯度"""
        x, y = int(position[0]), int(position[1])

        # 使用中心差分计算梯度
        if 0 < x < self.map_size[0] - 1 and 0 < y < self.map_size[1] - 1:
            grad_x = (self.total_field[x + 1, y] - self.total_field[x - 1, y]) / 2.0
            grad_y = (self.total_field[x, y + 1] - self.total_field[x, y - 1]) / 2.0
        else:
            grad_x, grad_y = 0, 0

        return np.array([grad_x, grad_y])

    def visualize_fields(self, path=None):
        """可视化势场和路径"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 引力场
        im1 = axes[0, 0].imshow(self.attractive_field.T, cmap='viridis', origin='lower')
        axes[0, 0].set_title('Attractive Potential Field')
        axes[0, 0].scatter(self.goal[0], self.goal[1], c='red', s=100, marker='*')
        plt.colorbar(im1, ax=axes[0, 0])

        # 斥力场
        im2 = axes[0, 1].imshow(self.repulsive_field.T, cmap='hot', origin='lower')
        axes[0, 1].set_title('Repulsive Potential Field')
        plt.colorbar(im2, ax=axes[0, 1])

        # 合势场
        im3 = axes[1, 0].imshow(self.total_field.T, cmap='plasma', origin='lower')
        axes[1, 0].set_title('Total Potential Field')
        axes[1, 0].scatter(self.goal[0], self.goal[1], c='red', s=100, marker='*')
        plt.colorbar(im3, ax=axes[1, 0])

        # 路径
        axes[1, 1].imshow(self.obstacle_map.T, cmap='gray', origin='lower')
        axes[1, 1].set_title('Planned Path')
        axes[1, 1].scatter(self.goal[0], self.goal[1], c='red', s=100, marker='*')

        if path is not None:
            axes[1, 1].plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
            axes[1, 1].scatter(path[0, 0], path[0, 1], c='green', s=100, marker='o')

        plt.tight_layout()
        plt.savefig('potential_fields.png', dpi=150, bbox_inches='tight')
        plt.show()