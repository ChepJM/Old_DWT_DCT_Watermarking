import copy
import numpy as np
import cv2
from scipy.stats.stats import pearsonr

# Преобразование Хаара
def HaarTransform(data, level):
    length = len(data)
    a = [0] * (length // 2)                 # Коэффициенты приближения
    d = [0] * (length // 2)                 # Коэффициенты детализации
    result = data

    for step in range(level):
        # Расчет коэффициентов
        for i in range(0, length // 2):
            a[i] = (result[2 * i] + result[2 * i + 1]) / 2
            d[i] = result[2 * i] - a[i]
        # Перенос коэффициентов в результирующий массив
        for i in range(0, length // 2):
            result[i] = a[i]
            result[i + length // 2] = d[i]
        length //= 2

# Преобразование Хаара для строки
def TransformRow(data, row_number, level, inverse=False):
    length = len(data[0])                   # Длина строки
    result = [0] * length                   # Массив для хранения преобразуемой строки

    for i in range(length):
        result[i] = data[row_number][i]

    if inverse:
        HaarTransformInverse(result, level)
    else:
        HaarTransform(result, level)

    for i in range(length):
        data[row_number][i] = result[i]

# Преобразование Хаара для столбца
def TransformColumn(data, column_number, level, inverse=False):
    length = len(data)                      # Длина столбца
    result = [0] * length                   # Массив для хранения преобразуемого столбца

    for i in range(length):
        result[i] = data[i][column_number]

    if inverse:
        HaarTransformInverse(result, level)
    else:
        HaarTransform(result, level)

    for i in range(length):
        data[i][column_number] = result[i]

# Дискретное вейвлет преобразование
def DWT(data):
    for i in range(data.shape[0]):
        TransformRow(data, i, 1)
    for i in range(data.shape[1]):
        TransformColumn(data, i, 1)

def IDWT(data):
    for i in range(data.shape[0]):
        TransformRow(data, i, 1, inverse=True)
    for i in range(data.shape[1]):
        TransformColumn(data, i, 1, inverse=True)

# Обратное преобразование Хаара
def HaarTransformInverse(data, level):
    result = data
    length = len(data)
    temp = [0] * length
    count = length // 2 ** level

    for step in range(level):
        for i in range(count):
            temp[2 * i] = result[i] + result[i + count]
            temp[2 * i + 1] = result[i] - result[i + count]
        for i in range(count * 2):
            result[i] = temp[i]
        count *= 2

# Преобразование Арнольда
def ArnoldTransform(data):
    temp = copy.deepcopy(data)
    N = len(data)

    for row_index in range(N):
        for column_index in range(N):
            new_row_index = (row_index + column_index) % N
            new_column_index = (row_index + 2 * column_index) % N
            data[new_row_index][new_column_index] = temp[row_index][column_index]

# Обратное преобразование Арнольда
def ArnoldTransformInverse(data):
    temp = copy.deepcopy(data)
    N = len(data)

    for row_index in range(N):
        for column_index in range(N):
            new_row_index = (2 * row_index - column_index) % N
            new_column_index = (column_index - row_index) % N
            data[new_row_index][new_column_index] = temp[row_index][column_index]

# Перевод двумерного массива в одномерный
def ToOneDim(data):
    temp = []
    for x in range(len(data)):
        for y in range(len(data[0])):
            temp.append(data[x][y])
    return temp

# Вспомогательная функция для дискретного косинусного преобразова-ния
def DCT_c(i, N):
    if i == 0:
        return np.sqrt(1 / N)
    else:
        return np.sqrt(2 / N)

# Дискретное косинусное преобразование
def DCT(data, M=8, N=8):
    temp = copy.deepcopy(data)
    temp_sum = 0

    for u in range(M):
        for v in range(N):
            for k in range(M):
                for l in range(N):
                    temp_sum += data[k][l] * np.cos((2*k + 1)*u*np.pi / (2 * M)) * np.cos((2*l + 1)*v*np.pi / (2 * N))
            temp[u][v] = DCT_c(u, M) * DCT_c(v, N) * temp_sum
            temp_sum = 0
    for x in range(M):
        for y in range(N):
            data[x][y] = temp[x][y]

# Обратное косинусное дискретное преобразование
def IDCT(data, M=8, N=8):
    temp = copy.deepcopy(data)
    temp_sum = 0

    for k in range(M):
        for l in range(N):
            for u in range(M):
                for v in range(N):
                    temp_sum += DCT_c(u, M) * DCT_c(v, N) * data[u][v] * np.cos((2*k + 1)*u*np.pi / (2 * M)) * np.cos((2*l + 1)*v*np.pi / (2 * N))
            temp[k][l] = temp_sum
            temp_sum = 0
    for x in range(M):
        for y in range(N):
            data[x][y] = temp[x][y]

# Функция генерации двух псевдослучайных последовательностей длины 64
def genSeq(seed):
    np.random.seed(seed)
    seq = np.random.randint(0, 2, 26)
    seq_0 = seq
    np.random.seed(seed + 5)
    seq = np.random.randint(0, 2, 26)
    seq_1 = seq
    return seq_0, seq_1

def encode():
    image = cv2.imread("C:/test.png")  # Считывание исходного изображения
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = gray_image.astype(float)
    DWT(gray_image)  # Применение дискретного вейвлет преобразования

    # Разбиение LL поддиапазона на блоки 8x8 и применение к ним дис-кретного косинусного преобразования
    temp = [[0 for x in range(8)] for y in range(8)]
    for width in range(0, gray_image.shape[0] // 2, 8):
        for heigth in range(0, gray_image.shape[1] // 2, 8):
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    temp[i - width][j - heigth] = gray_image[i][j]
            DCT(temp)
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    gray_image[i][j] = temp[i - width][j - heigth]

    # Предобработка водяного знака
    water_image = cv2.imread("C:/water2.png", 2)  # Считывание водяного знака
    ret, water_image = cv2.threshold(water_image, 127, 1, cv2.THRESH_BINARY)
    for i in range(10):
        ArnoldTransform(water_image)  # 10 итераций преобразования Арнольда

    key = 12345
    seq_0, seq_1 = genSeq(key)
    seq_0 = list(map({0: 1, 1: -1}.get, seq_0))
    seq_1 = list(map({0: 1, 1: -1}.get, seq_1))

    alpha = 5
    indexes = [[0, 7], [1, 6], [1, 7], [2, 5], [2, 6], [2, 7], [3, 4], [3, 5], [3, 6], [3, 7],
               [4, 3], [4, 4], [4, 5], [4, 6], [5, 2], [5, 3], [5, 4], [5, 5], [6, 1], [6, 2],
               [6, 3], [6, 4], [7, 0], [7, 1], [7, 2], [7, 3]]
    for i in range(32):
        for j in range(32):
            if water_image[i][j]:
                for k in range(26):
                    gray_image[i * 8 + indexes[k][0]][j * 8 + indexes[k][1]] += alpha * seq_1[k]
            else:
                for k in range(26):
                    gray_image[i * 8 + indexes[k][0]][j * 8 + indexes[k][1]] += alpha * seq_0[k]


    for width in range(0, gray_image.shape[0] // 2, 8):
        for heigth in range(0, gray_image.shape[1] // 2, 8):
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    temp[i - width][j - heigth] = gray_image[i][j]
            IDCT(temp)
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    gray_image[i][j] = temp[i - width][j - heigth]

    IDWT(gray_image)
    gray_image = gray_image.astype(np.uint8)
    cv2.imwrite("C:/test.png", gray_image)  # Сохранение изображения

def decode():
    gray_image = cv2.imread("C:/test.png", 2)
    gray_image = gray_image.astype(float)
    DWT(gray_image)  # Применение дискретного вейвлет преобразования

    # Разбиение LL поддиапазона на блоки 8x8 и применение к ним дис-кретного косинусного преобразования
    temp = [[0 for x in range(8)] for y in range(8)]
    for width in range(0, gray_image.shape[0] // 2, 8):
        for heigth in range(0, gray_image.shape[1] // 2, 8):
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    temp[i - width][j - heigth] = gray_image[i][j]
            DCT(temp)
            for i in range(width, width + 8):
                for j in range(heigth, heigth + 8):
                    gray_image[i][j] = temp[i - width][j - heigth]

    key = 12345
    seq_0, seq_1 = genSeq(key)

    #alpha = 0.5
    indexes = [[0, 7], [1, 6], [1, 7], [2, 5], [2, 6], [2, 7], [3, 4], [3, 5], [3, 6], [3, 7],
               [4, 3], [4, 4], [4, 5], [4, 6], [5, 2], [5, 3], [5, 4], [5, 5], [6, 1], [6, 2],
               [6, 3], [6, 4], [7, 0], [7, 1], [7, 2], [7, 3]]
    dec_im = np.ndarray(shape=(32, 32), dtype=np.uint8)
    temp2 = [0 for x in range(26)]
    for i in range(32):
        for j in range(32):
            for k in range(26):
                temp2[k] = gray_image[i * 8 + indexes[k][0]][j * 8 + indexes[k][1]]
            if abs(np.corrcoef(temp2, seq_0)[0][1]) > abs(np.corrcoef(temp2, seq_1)[0][1]):
                dec_im[i][j] = 0
            else:
                dec_im[i][j] = 1

    for i in range(10):
        ArnoldTransformInverse(dec_im)
    ret, dec_im = cv2.threshold(dec_im, 0, 255, cv2.THRESH_BINARY)

    cv2.imwrite("C:/test.png", dec_im)

encode()
decode()
