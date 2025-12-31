import pygame
import numpy as np
import sys


class PyGameSimulation:


    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("APF Obstacle Avoidance Simulation")

        # 颜色定义
        self.colors = {
            'background': (255, 255, 255),
            'robot': (0, 100, 255),
            'obstacle': (200, 50, 50),
            'goal': (50, 200, 50),
            'path': (0, 0, 0),
            'text': (0, 0, 0)
        }

        # 仿真参数
        self.robot_radius = 15
        self.robot_pos = np.array([100, 300])
        self.goal_pos = np.array([700, 300])
        self.obstacles = []

        # 物理参数
        self.max_speed = 3.0
        self.dt = 0.1

        # APF规划器
        self.apf = ArtificialPotentialField(map_size=(width, height))

        # 路径记录
        self.path_history = []

    def add_obstacle(self, center, radius):
        """添加障碍物"""
        self.obstacles.append({'center': np.array(center), 'radius': radius})

    def create_obstacle_map(self):
        """创建障碍物地图"""
        obstacle_map = np.zeros((self.width, self.height))

        for obs in self.obstacles:
            center = obs['center'].astype(int)
            radius = obs['radius']

            # 在障碍物位置标记为1
            for x in range(max(0, center[0] - radius), min(self.width, center[0] + radius)):
                for y in range(max(0, center[1] - radius), min(self.height, center[1] + radius)):
                    if np.linalg.norm([x - center[0], y - center[1]]) <= radius:
                        obstacle_map[x, y] = 1

        return obstacle_map

    def compute_apf_force(self):
        """计算APF控制力"""
        # 更新APF参数
        obstacle_map = self.create_obstacle_map()
        self.apf.set_goal(self.goal_pos)
        self.apf.set_obstacles(obstacle_map)
        self.apf.compute_total_field()

        # 计算当前位置的梯度
        gradient = self.apf.compute_gradient(self.robot_pos)

        # 转换为控制力
        if np.linalg.norm(gradient) > 0:
            force = -gradient / np.linalg.norm(gradient) * self.max_speed
        else:
            force = np.zeros(2)

        return force

    def update_robot(self):
        """更新机器人位置"""
        force = self.compute_apf_force()

        # 更新位置
        self.robot_pos += force * self.dt

        # 边界检查
        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.width)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.height)

        # 记录路径
        self.path_history.append(self.robot_pos.copy())

    def draw_scene(self):
        """绘制场景"""
        self.screen.fill(self.colors['background'])

        # 绘制障碍物
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, self.colors['obstacle'],
                               obs['center'].astype(int), obs['radius'])

        # 绘制目标
        pygame.draw.circle(self.screen, self.colors['goal'],
                           self.goal_pos.astype(int), 20)

        # 绘制机器人
        pygame.draw.circle(self.screen, self.colors['robot'],
                           self.robot_pos.astype(int), self.robot_radius)

        # 绘制路径
        if len(self.path_history) > 1:
            pygame.draw.lines(self.screen, self.colors['path'], False,
                              [p.astype(int) for p in self.path_history], 2)

        # 显示信息
        font = pygame.font.Font(None, 36)
        distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        text = font.render(f"Distance to goal: {distance:.1f}", True, self.colors['text'])
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def run_simulation(self, max_steps=1000):
        """运行仿真"""
        clock = pygame.time.Clock()
        running = True
        step = 0

        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 点击添加障碍物
                    mouse_pos = pygame.mouse.get_pos()
                    self.add_obstacle(mouse_pos, 30)

            # 检查是否到达目标
            if np.linalg.norm(self.robot_pos - self.goal_pos) < 20:
                print("Goal reached!")
                running = False

            # 更新和绘制
            self.update_robot()
            self.draw_scene()

            clock.tick(60)  # 60 FPS
            step += 1

        pygame.quit()
        sys.exit()