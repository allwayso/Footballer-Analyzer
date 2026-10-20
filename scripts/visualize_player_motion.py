"""从已有运行生成球员轨迹回放、活动热图和跑动对比独立页。"""

import argparse

from footballer_analyzer.motion_visualization import create_motion_visualization


def main():
    """解析运行目录并打印可通过现有服务打开的报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="已完成人工绑定和场地标定的运行目录")
    args = parser.parse_args()
    print(create_motion_visualization(args.run))


if __name__ == "__main__":
    main()
