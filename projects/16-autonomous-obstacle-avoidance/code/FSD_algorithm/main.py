import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description='FSD_algorithm-style Obstacle Avoidance System')
    parser.add_argument('--mode', choices=['perception', 'planning', 'simulation'],
                        default='simulation', help='运行模式')
    parser.add_argument('--image_path', type=str, default='data/test_image.jpg',
                        help='测试图像路径')
    parser.add_argument('--camera_params', type=float, nargs=4,
                        default=[525.0, 525.0, 320.0, 240.0],
                        help='相机参数 [fx, fy, cx, cy]')

    args = parser.parse_args()

    if args.mode == 'perception':
        # 感知模块演示
        print("运行感知模块...")

        # 初始化感知模块
        occupancy_gen = PseudoOccupancyGenerator()
        bev_projector = SimpleBEVProjector()

        # 读取图像
        image = cv2.imread(args.image_path)
        if image is None:
            print(f"无法读取图像: {args.image_path}")
            return

        # 生成3D占用网格
        voxel_grid = occupancy_gen.process_frame(image, args.camera_params)
        print(f"生成3D占用网格，形状: {voxel_grid.shape}")

        # 生成BEV地图
        bev_map = bev_projector.voxel_to_bev(voxel_grid)

        # 可视化
        bev_vis = bev_projector.visualize_bev(bev_map)
        cv2.imshow('BEV Map', bev_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'planning':
        # 规划模块演示
        print("运行规划模块...")

        # 创建测试地图
        map_size = (100, 100)
        obstacle_map = np.zeros(map_size)

        # 添加一些障碍物
        obstacle_map[30:50, 30:50] = 1  # 方形障碍物
        obstacle_map[60:80, 20:40] = 1  # 另一个障碍物

        # 初始化APF
        apf = ArtificialPotentialField(map_size=map_size)
        apf.set_goal([90, 90])
        apf.set_obstacles(obstacle_map)
        apf.compute_total_field()

        # 规划路径
        path = apf.plan_path([10, 10])

        # 可视化
        apf.visualize_fields(path)

    elif args.mode == 'simulation':
        # 仿真演示
        print("运行仿真...")

        sim = PyGameSimulation()

        # 添加一些障碍物
        sim.add_obstacle([400, 200], 40)
        sim.add_obstacle([300, 400], 30)
        sim.add_obstacle([500, 350], 35)

        # 运行仿真
        sim.run_simulation()

    print("程序执行完成！")


if __name__ == "__main__":
    main()