import matching
import cv2
import numpy as np
import ctypes
import win32api
import win32gui
import win32con
import torch
from torchvision import models, transforms
import os
from PIL import ImageGrab, Image
import time
from config import *
from scipy.spatial.distance import cosine
import shutil

# 加载预训练的ResNet模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()  # 设为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取图像特征
def extract_features(image_ndarray: np.ndarray):
    """
    image_ndarray: 通过OpenCV读取的numpy.ndarray图像(BGR通道)
    """
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2RGB)
    # 转成 PIL Image
    pil_image = Image.fromarray(image_rgb)
    # 再做transform预处理
    tensor_image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor_image)
    return features.cpu().numpy().flatten()

# 计算余弦相似度
def calculate_similarity(features1, features2):
    return 1 - cosine(features1, features2)

# 判断两张图片是否相似
def are_images_similar(image1, image2, threshold=0.90):
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    similarity = calculate_similarity(features1, features2)
    return similarity > threshold

# 获取窗体坐标位置(左上)
def getGameWindowPosition():
    # FindWindow(lpClassName=None, lpWindowName=None)  窗口类名 窗口标题名
    window = win32gui.FindWindow(None,WINDOW_TITLE)
    # 没有定位到游戏窗体
    while not window:
        print('定位游戏窗体失败，5秒后重试...')
        time.sleep(5)
        window = win32gui.FindWindow(None,WINDOW_TITLE)
    # 定位到游戏窗体
    win32gui.SetForegroundWindow(window) # 将窗体顶置
    # 将窗体移动到(0,0)
    # win32gui.MoveWindow(window, 0, 0, width, height, True)
    pos = win32gui.GetWindowRect(window)
    print("定位到游戏窗体：" + str(pos))
    client_rect = win32gui.GetClientRect(window)
    print(f"定位到游戏窗体：{pos}, 客户区大小：{client_rect}")
    return (pos[0],pos[1])

# 获取一张完整的屏幕截图
def getScreenImage():
    print('捕获屏幕截图...')
    scim = ImageGrab.grab()  # 屏幕截图，获取到的是Image类型对象
    scim.save('screen.png')
    return cv2.imread("screen.png") # opencv 读取，拿到的是ndarray存储的图像

# 从屏幕截图中识别
def getAllSquare(screen_image,game_pos):
    # 确保文件夹存在
    if not os.path.exists('slice'):
        os.makedirs('slice')
    print('图像切片处理...')
    # 通过游戏窗体，找到连连看连接的区域：
    game_x = game_pos[0] + MARGIN_LEFT
    game_y = game_pos[1] + MARGIN_HEIGHT
    # 从连接区域左上开始，把图像切割成一个一个的小块，切割标准是按照小块的横纵坐标。
    all_square = []
    for x in range(0,H_NUM):
        # line_square = []
        for y in range(0,V_NUM):
            # ndarray的切片方法，[纵坐标起始位置：纵坐标结束为止，横坐标起始位置：横坐标结束位置]
            square = screen_image[game_y + y * SQUARE_HEIGHT :game_y + (y+1) * SQUARE_HEIGHT,game_x + x * SQUARE_WIDTH:game_x + (x+1) * SQUARE_WIDTH]
            # 保存每个方块图像到temp文件夹
            cut_square = square[SUB_LT_Y:SUB_RB_Y, SUB_LT_X:SUB_RB_X]
            square_filename = f'slice/square_{x}_{y}.png'  # 使用坐标(x, y)作为文件名
            cv2.imwrite(square_filename, cut_square)  # 保存图像
            all_square.append(square)
    # 因为有些图片的边缘不一致造成干扰（主要是空白区域的切图），所以把每张小方块向内缩小一部分再
    # 对所有的方块进行处理屏蔽掉外边缘 然后返回
    # return list(map(lambda square : square[SUB_LT_Y:SUB_RB_Y,SUB_LT_X:SUB_RB_X],all_square))
    return [square[SUB_LT_Y:SUB_RB_Y, SUB_LT_X:SUB_RB_X] for square in all_square]
    # 上面这行相当于下面这4行
    # new_all_square = []
    # for square in all_square:
    #     s = square[SUB_LT_Y:SUB_RB_Y, SUB_LT_X:SUB_RB_X]
    #     new_all_square.append(s)
    # return new_all_square

# 判断图像是否与已经在列表中的图像相同，如果是返回True
def isImageExist(img,img_list, threshold=0.80):
    for existed_img in img_list:
        # 每组方块图像完全相同时使用此代码
        # b = np.subtract(existed_img,img) # 图片数组进行比较，返回的是两个图片像素点差值的数组，
        # if not np.any(b):   # 如果全部是0，说明两图片完全相同。
        #     return True
        # else:
        #     continue
        # 有些情况下，每组图片有微小的差异，色值或错位等，这时使用相似度判断，可以容忍微笑的错位及色值变化
        if are_images_similar(existed_img, img):
            return True
    return False

# 获取所有的方块类型
def getAllSquareTypes(all_square):
    # 如果文件夹存在，则删除并重新创建
    if os.path.exists('temp'):
        shutil.rmtree('temp')  # 删除文件夹及其中所有内容
    os.makedirs('temp')  # 重新创建文件夹
    n = 1
    print("将图像矩阵按类型归入类型列表...")
    types = []
    # 先把空白添加到数组中，作为0号
    empty_img = cv2.imread('empty.png')
    types.append(empty_img)
    for square in all_square:
        # 如果这个图像不存在的话将图像保存起来
        if not isImageExist(square,types):
            # 保存每个方块图像到temp文件夹
            square_filename = f'temp/square_{n}.png'  # 使用坐标(x, y)作为文件名
            cv2.imwrite(square_filename, square)  # 保存图像
            n = n + 1
            types.append(square)
    return types

# 将所有的方块与类型进行比较，转置成数字
def getAllSquareRecord(all_square_list,types):
    print("将所有的方块与类型进行比较，转置成数字矩阵...")
    record = []  # 整个记录的二维数组
    line = []   # 记录一行
    for square in all_square_list:   # 把所有的方块和保存起来的所有类型做对比
        num = 0
        for type in types:    # 所有类型
            # 判断完全一样
            # res = cv2.subtract(square,type) # 作比较
            # if not np.any(res):     # 如果两个图片一样
            # 使用模糊判断是否匹配，能够容忍微笑的错位及颜色变化
            if are_images_similar(square, type):
                line.append(num)    # 将类型的数字记录进这一行
                break               # 并且跳出循环
            num += 1                # 如果没有匹配上，则类型数加1

        if len(line) == V_NUM:         # 如果校验完这一行已经有了11个数据，则另起一行
            record.append(line)
            line = []
    print(record)
    return record

# 自动消除
def autoRelease(result,game_x,game_y):
    for i in range(0,len(result)):
        for j in range(0,len(result[0])):
            # 以上两个for循环，定位第一个选中点
            if result[i][j] != 0:
                for m in range(0,len(result)):
                    for n in range(0,len(result[0])):
                        if result[m][n] != 0:
                            # 后两个for循环定位第二个选中点
                            if matching.canConnect(i,j,m,n,result):
                            # 执行消除算法并返回
                                result[i][j] = 0
                                result[m][n] = 0
                                x1 = game_x + j*SQUARE_WIDTH
                                y1 = game_y + i*SQUARE_HEIGHT
                                x2 = game_x + n*SQUARE_WIDTH
                                y2 = game_y + m*SQUARE_HEIGHT
                                click1X = int((x1 + 15)/DISPLAY_SCALING)
                                click2Y = int((y1 + 18)/DISPLAY_SCALING)
                                win32api.SetCursorPos((click1X,click2Y))
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, click1X, click2Y, 0, 0)
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, click1X, click2Y, 0, 0)
                                time.sleep(TIME_INTERVAL)

                                click2X = int((x2 + 15)/DISPLAY_SCALING)
                                click2Y = int((y2 + 18)/DISPLAY_SCALING)
                                win32api.SetCursorPos((click2X, click2Y))
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, click2X, click2Y, 0, 0)
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, click2X, click2Y, 0, 0)
                                time.sleep(TIME_INTERVAL)
                                print('可消除点：'+ str(i+1) + ',' + str(j+1) + '(' + str(click1X) + ',' + str(click2Y) +')和' + str(m+1) + ',' + str(n+1) + '(' + str(click2X) + ',' + str(click2Y) + ')')
                                return True
    return False

def autoRemove(squares,game_pos):
    # 每次消除一对儿，QQ的连连看最多105对儿
    game_x = game_pos[0] + MARGIN_LEFT
    game_y = game_pos[1] + MARGIN_HEIGHT
    # 判断是否消除完了？如果没有的话，点击重列后继续消除
    for i in range(0,105):
        autoRelease(squares,game_x,game_y)


if __name__ == '__main__':
    # 1、定位游戏窗体
    game_pos = getGameWindowPosition()
    time.sleep(1)
    # 2、从屏幕截图一张，通过opencv读取
    screen_image = getScreenImage()
    # 3、图像切片，把截图中的连连看切成一个一个的小方块，保存在一个数组中
    all_square_list = getAllSquare(screen_image,game_pos)
    # 4、切片处理后的图片，相同的作为一种类型，放在数组中。
    types = getAllSquareTypes(all_square_list)
    # 5、将切片处理后的图片，转换成相对应的数字矩阵。注意 拿到的数组是横纵逆向的，转置一下。
    result = np.transpose(getAllSquareRecord(all_square_list,types))
    # 6、执行自动消除
    autoRemove(result,game_pos)
    # 7、消除完成，释放资源。
    cv2.waitKey(0)
    cv2.destroyAllWindows()







