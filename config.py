# -*- coding:utf-8 -*-
__author__ = 'CrazyBunQnQ'
__Date__ = '2024/11/12 13:14'

# 窗体标题  用于定位游戏窗体
WINDOW_TITLE = "BD2"
# 时间间隔  间隔多少秒连一次
TIME_INTERVAL = 0.3
# 游戏区域距离顶点的长度
MARGIN_LEFT = 353
# 游戏区域距离顶点的高度
MARGIN_HEIGHT = 271
# 横向的方块数量
H_NUM = 16
# 纵向的方块数量
V_NUM = 7
# 方块宽度
SQUARE_WIDTH = 77
# 方块高度
SQUARE_HEIGHT = 93
# 切片处理时候的左上、右下坐标：
# 注意  这里要么保证裁剪后的像素是21*25，要么（比如四个数据是10,10,50,50；也就是40*40像素）把empty.png图片替换成对应大小的一张图片（比如40*40），图片可以没用，但程序中不能
SUB_LT_X = 10
SUB_LT_Y = 11
SUB_RB_X = 50
SUB_RB_Y = 71
# 万恶的 Windows 缩放比例...
DISPLAY_SCALING = 1.5