import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pygame
import sys
def insidetriangle(xList, yList):
    xs = xList
    ys = yList
    x_range = np.arange(np.min(xs), np.max(xs) + 1)
    y_range = np.arange(np.min(ys), np.max(ys) + 1)

    X, Y = np.meshgrid(x_range, y_range)
    xc = np.mean(xs)
    yc = np.mean(ys)

    triangle = np.ones(X.shape, dtype=bool)
    for i in range(3):
        ii = (i + 1) % 3
        if xs[i] == xs[ii]:
            include = X * (xc - xs[i]) / abs(xc - xs[i]) > xs[i] * (xc - xs[i]) / abs(xc - xs[i])
        else:
            poly = np.poly1d([(ys[ii] - ys[i]) / (xs[ii] - xs[i]), ys[i] - xs[i] * (ys[ii] - ys[i]) / (xs[ii] - xs[i])])
            include = Y * (yc - poly(xc)) / abs(yc - poly(xc)) > poly(X) * (yc - poly(xc)) / abs(yc - poly(xc))
        triangle *= include

    return X[triangle], Y[triangle]


XListFir = [1 + 90, 300 + 50, 50]
YListFir = [10, 50, 150]
ZListFir = [50, 50, -90]

XListSec = [1, 135, 100]
YListSec = [25, 60, 250]
ZListSec = [5, 5, -20]

x1 = XListFir[:]
x1.append(XListFir[0])
y1 = YListFir[:]
y1.append(YListFir[0])
z1 = ZListFir[:]
z1.append(ZListFir[0])

x2 = XListSec[:]
x2.append(XListSec[0])
y2 = YListSec[:]
y2.append(YListSec[0])
z2 = ZListSec[:]
z2.append(ZListSec[0])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('axis X')
ax.set_ylabel('axis Y')
ax.set_zlabel('axis Z')
ax.plot(x1,y1,z1)
ax.plot(x2,y2,z2)

massX, massY = insidetriangle(XListFir, YListFir)
massX2, massY2 = insidetriangle(XListSec, YListSec)

'''
massX, massY = bufferFirstTriangle[0], bufferFirstTriangle[1]
massX2 = bufferSecondTriangle[0]
massY2 = bufferSecondTriangle[1]
'''
print(massX)
print(massY)


coord = [
    [XListFir[0], YListFir[0], 1],
    [XListFir[1], YListFir[1], 1],
    [XListFir[2], YListFir[2], 1],
    ZListFir]

coord2 = [
    [XListSec[0], YListSec[0], 1],
    [XListSec[1], YListSec[1], 1],
    [XListSec[2], YListSec[2], 1],
    ZListSec ]
print(coord)


def ploskost(coord):
    M5 = np.array([coord[0], coord[1], coord[2]])
    v5 = np.array(coord[3])
    result = np.linalg.solve(M5, v5)
    return [round(elem, 2) for elem in result]

urPloskosti = ploskost(coord)
urPloskosti2 = ploskost(coord2)
print(urPloskosti)


def getZcoord(x, y, urPloskosti):
    return float(urPloskosti[0] * x + urPloskosti[1] * y + urPloskosti[2])

WIDTH = 500
pygame.init()
sc = pygame.display.set_mode((900, 500))


zbuff = [[-10000] * WIDTH for i in range(WIDTH)]  #
pygame.time.delay(0)
for i in range(len(massX)):
    if (getZcoord(massX[i], massY[i], urPloskosti) > zbuff[int(massX[i])][int(massY[i])]):
        zbuff[int(massX[i])][int(massY[i])] = 1.

for i in range(len(massX2)):
    if (getZcoord(massX2[i], massY2[i], urPloskosti2) > zbuff[int(massX2[i])][int(massY2[i])]):
        zbuff[int(massX2[i])][int(massY2[i])] = 2.
sc.fill((200,200,255))
for i in range(len(zbuff)):
    for j in range(len(zbuff)):
        if (zbuff[i][j] == 1):
            pygame.draw.aaline(sc, (78, 90, 89), [i, j], [i, j], 9)
        if (zbuff[i][j] == 2):
            pygame.draw.aaline(sc, (56, 89, 25), [i, j], [i, j], 18)

pygame.display.update()
FPS = 60
clock = pygame.time.Clock()
while True:

    # задержка
    clock.tick(FPS)

    # цикл обработки событий
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            exit()

    # --------
    # изменение объектов и многое др.
    # --------

    # обновление экрана
    pygame.display.update()
