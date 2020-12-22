import copy
import glob
from os.path import join
import numpy as np
from dicAOPC.ImageDataGenerator_GazeCapture import ImageDataGenerator
import tensorflow as tf
import global_variables as var
import cv2
import time
import datetime
import copy
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from dicAOPC.preProcessing import preprocessITracker_Keras

table = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# これはいつか汎用化のために整理する
heatmapSize = 12
ImgSize = 128


# 可視化技術を評価する手法
# img_path : 画像の位置，サブディレクトリも全て捜索
# model_path : 使用したモデル
# flag : Grad-CAMを使用するか，既に作成済みの画像を使うか．true:使う false:作成済み
def AOPC_GradCAM_Random(img_path, model_path, flag):
    face_dir = "/mnt/data2/img/20200209/face"
    right_dir = "/mnt/data2/img/20200209/right_eye"
    left_dir = "/mnt/data2/img/20200209/left_eye"
    grid_dir = "/mnt/data2/img/20200209/grid"

    # モデルの読み込み
    model = tf.keras.models.load_model(model_path)
    seqs_face = sorted(glob.glob(join(face_dir, "*")))
    seqs_right = sorted(glob.glob(join(right_dir, "*")))
    seqs_left = sorted(glob.glob(join(left_dir, "*")))
    seqs_grid = sorted(glob.glob(join(grid_dir, "*")))

    gradCamPath = "/mnt/data2/img/20200209/"
    test = np.load("/mnt/data2/img/20200209/grad_cam_heatmap.npy")
    heatmaps = ReshapeNdarray(gradCamPath)

    batchSize = 40

    j = 0

    data_gen = ImageDataGenerator()
    for batch, heatmap in data_gen.flow(seqs_face, seqs_right, seqs_left, seqs_grid, heatmaps, 128, 128, 3, batchSize):

        # batchAOPC: batchSize, 可視化手法(4), 入力パターン(7), AOPC(144), xy(2)
        batchAOPC = np.zeros((batch[0].shape[0], 7, 144, 2))

        # 摂動後の画像は４つの可視化手法それぞれで摂動を行う．
        perturbationImg = copy.deepcopy(batch)

        predictValues = model.predict_on_batch(batch)

        # 摂動の回数だけ実行．今回は画像全体で行うとして画像サイズの12*12でやっている．後で変数に変更
        for i in range(12 * 12):

            start = time.time()

            # 摂動した画像を作成する関数
            perturbationImg = makePerturbationImg_noPrepro_random(perturbationImg)

            preprocessImg = preprocessITracker_Keras(perturbationImg)

            # オリジナル画像と摂動画像からAOPCを算出する関数
            batchAOPC[:, :, i, :] = calculateAOPCrandom(
                preprocessImg, batch, model, predictValues)

            process_time = time.time() - start
            # print("perturbation:" + str(process_time))

        if j == 0:
            allAopcValues = np.copy(batchAOPC)
            j = 1
        else:
            allAopcValues = np.concatenate([allAopcValues, batchAOPC], axis=0)

        # now = datetime.datetime.now()
        # np.save(gradCamPath + "/aopc/aopc_random_" + now.strftime('%Y%m%d_%H%M%S'), allAopcValues)
    now = datetime.datetime.now()
    np.save(gradCamPath + "/aopc/aopc_random_" + now.strftime('%Y%m%d_%H%M%S'), allAopcValues)

    return


# 可視化技術を評価する手法
# img_path : 画像の位置，サブディレクトリも全て捜索
# model_path : 使用したモデル
# flag : Grad-CAMを使用するか，既に作成済みの画像を使うか．true:使う false:作成済み
def AOPC_GradCAM(img_path, model_path, flag):
    face_dir = "/mnt/data2/img/20200209/face"
    right_dir = "/mnt/data2/img/20200209/right_eye"
    left_dir = "/mnt/data2/img/20200209/left_eye"
    grid_dir = "/mnt/data2/img/20200209/grid"

    # モデルの読み込み
    model = tf.keras.models.load_model(model_path)
    seqs_face = sorted(glob.glob(join(face_dir, "*")))
    seqs_right = sorted(glob.glob(join(right_dir, "*")))
    seqs_left = sorted(glob.glob(join(left_dir, "*")))
    seqs_grid = sorted(glob.glob(join(grid_dir, "*")))

    gradCamPath = "/mnt/data2/img/20200209/"
    test = np.load("/mnt/data2/img/20200209/grad_cam_heatmap.npy")
    heatmaps = ReshapeNdarray(gradCamPath)

    batchSize = 40

    j = 0

    data_gen = ImageDataGenerator()
    for batch, heatmap in data_gen.flow(seqs_face, seqs_right, seqs_left, seqs_grid, heatmaps, 128, 128, 3, batchSize):

        # batchAOPC: batchSize, 可視化手法(4), 入力パターン(7), AOPC(144), xy(2)
        batchAOPC = np.zeros((batch[0].shape[0], 4, 7, 144, 2))

        # 摂動後の画像は４つの可視化手法それぞれで摂動を行う．
        perturbationImg = [copy.deepcopy(batch), copy.deepcopy(batch), copy.deepcopy(batch), copy.deepcopy(batch)]

        predictValues = model.predict_on_batch(batch)

        # 摂動の回数だけ実行．今回は画像全体で行うとして画像サイズの12*12でやっている．後で変数に変更
        for i in range(12 * 12):

            start = time.time()

            # 摂動した画像を作成する関数
            perturbationImg = makePerturbationImg_noPrepro(heatmap[:, :, :, i, :], perturbationImg)

            preprocessImg = []
            for img in perturbationImg:
                preprocessImg.append(preprocessITracker_Keras(img))

            # オリジナル画像と摂動画像からAOPCを算出する関数
            batchAOPC[:, :, :, i, :] = calculateAOPC(
                preprocessImg, batch, model, predictValues)

            process_time = time.time() - start
            # print("perturbation:" + str(process_time))

        if j == 0:
            allAopcValues = np.copy(batchAOPC)
            j = 1
        else:
            allAopcValues = np.concatenate([allAopcValues, batchAOPC], axis=0)

        # now = datetime.datetime.now()
        # np.save(gradCamPath + "/aopc/aopc_" + now.strftime('%Y%m%d_%H%M%S'), allAopcValues)
    now = datetime.datetime.now()
    np.save(gradCamPath + "/aopc/aopc_" + now.strftime('%Y%m%d_%H%M%S'), allAopcValues)

    return


# ヒートマップの情報とバッチからバッチと同じサイズの摂動後のリストを返す関数
# heatmap[x,画像のタイプ，使用するメソッド，1,3]
# prebatch 4つのndarrayを持つリストが４つ繋がったリスト　画像が格納されている
def makePerturbationImg_noPrepro_random(preBatch):
    start = time.time()

    # どの範囲でperturbationをするか決定する
    perturbationSize = int(ImgSize / heatmapSize)

    batch = copy.deepcopy(preBatch)

    # 目，顔ごとに処理する
    for i, img_array in enumerate(batch):

        if i != 3:
            # ランダムに初期化するための画像
            randomImg = np.random.randint(0, 255, (img_array.shape[0], perturbationSize, perturbationSize, 3))
            # randomImg = randomImg.astype('float32') / 255.

            random = np.random.randint(0, heatmapSize, (img_array.shape[0], 2))

            # ヒートマップの位置を示している．shape[batch,2]
            perturbationPosition = (random * perturbationSize).astype(np.int)

            for b in range(img_array.shape[0]):
                # aaaa = batch[m][i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                #       perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :]
                # print(aaaa.shape)
                # bbbb = randomImg[b, :]
                # print(bbbb.shape)
                batch[i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :] = randomImg[b, :]

            # batch[m][i][:, perturbationPosition[0]:perturbationPosition[0] + perturbationSize,
            # perturbationPosition[1]:perturbationPosition[1] + perturbationSize, ] = img_array[]

            # heatmap[:, :, h, :, :]

    # print("batch")
    process_time = time.time() - start
    # print("perturbation:" + str(process_time))
    return batch


# ヒートマップの情報とバッチからバッチと同じサイズの摂動後のリストを返す関数
# heatmap[x,画像のタイプ，使用するメソッド，1,3]
# prebatch 4つのndarrayを持つリストが４つ繋がったリスト　画像が格納されている
def makePerturbationImg_noPrepro(heatmap, preBatch):
    start = time.time()

    # どの範囲でperturbationをするか決定する
    perturbationSize = int(ImgSize / heatmapSize)

    batch = copy.deepcopy(preBatch)

    # 可視化手法ごとに取り出す．x,y,x+y,custom
    for m, method_list in enumerate(batch):

        # 目，顔ごとに処理する
        for i, img_array in enumerate(method_list):

            if i != 3:
                # ランダムに初期化するための画像
                randomImg = np.random.randint(0, 255, (img_array.shape[0], perturbationSize, perturbationSize, 3))
                # randomImg = randomImg.astype('float32') / 255.

                # ヒートマップの位置を示している．shape[batch,2]
                perturbationPosition = (heatmap[:, i, m, 1:] * perturbationSize).astype(np.int)

                for b in range(batch[m][i].shape[0]):
                    # aaaa = batch[m][i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                    #       perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :]
                    # print(aaaa.shape)
                    # bbbb = randomImg[b, :]
                    # print(bbbb.shape)
                    batch[m][i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                    perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :] = randomImg[b, :]

                # batch[m][i][:, perturbationPosition[0]:perturbationPosition[0] + perturbationSize,
                # perturbationPosition[1]:perturbationPosition[1] + perturbationSize, ] = img_array[]

                # heatmap[:, :, h, :, :]

    # print("batch")
    process_time = time.time() - start
    # print("perturbation:" + str(process_time))
    return batch


# 可視化技術を評価する手法
# img_path : 画像の位置，サブディレクトリも全て捜索
# model_path : 使用したモデル
# flag : Grad-CAMを使用するか，既に作成済みの画像を使うか．true:使う false:作成済み
def AOPC(img_path, model_path, flag):
    # 個人ディレクトリのリスト
    seqs = sorted(glob.glob(join(img_path, "0*")))

    # 使用モデル
    model = tf.keras.models.load_model(var.model_path)
    model.summary()

    # 個人の画像ディレクトリへアクセス
    for b, dir1 in enumerate(seqs):

        print("ディレクトリの現在の位置:" + dir1)

        # data generatorの作成(batch事に画像データを取ってくる)
        train_datagen = ImageDataGenerator()

        # 個人の画像ディレクトリの全ての可視化結果が格納されている
        # heatmap: x, 3, 4, 144, 3
        heatmap = ReshapeNdarray(dir1)

        # 個人ディレクトリの画像の現在の位置
        heatmap_count = 0

        # 個人それぞれのAOPC
        # personalAOPC: batchSize, 可視化手法(4), 入力パターン(7), AOPC(144), xy(2)
        personalAOPC = np.zeros((len(sorted(glob.glob(join(dir1, "0*")))), 4, 7, 144, 2))

        # バッチごとに処理
        # ndarrayが4つ入ったリスト
        # (目，目，顔，グリッド) それぞれバッチサイズ分格納
        for batch in train_datagen.flow_from_personal_directory(dir1, 128, 128, 3):

            # 摂動後の画像は４つの可視化手法それぞれで摂動を行う．
            perturbationImg = [copy.deepcopy(batch), copy.deepcopy(batch), copy.deepcopy(batch), copy.deepcopy(batch)]

            # 摂動の回数だけ実行．今回は画像全体で行うとして画像サイズの12*12でやっている．後で変数に変更
            for i in range(12 * 12):
                # 摂動した画像を作成する関数
                perturbationImg = makePerturbationImg(heatmap[heatmap_count:, :, :, i, :], perturbationImg)

                # オリジナル画像と摂動画像からAOPCを算出する関数
                personalAOPC[heatmap_count:(heatmap_count + batch[0].shape[0]), :, :, i, :] = calculateAOPC(
                    perturbationImg, batch, model)

            # print("heatmap:" + str(heatmap_count) +"\nbatchshape:" + str(batch[0].shape[0]))
            # print("shape" + str(personalAOPC[heatmap_count:(heatmap_count + batch[0].shape[0]), :, :, 0, :].shape))
            # 次のフレームの開始位置を保存
            heatmap_count = heatmap_count + batch[0].shape[0]

        now = datetime.datetime.now()
        np.save(dir1 + "/aopc_" + now.strftime('%Y%m%d_%H%M%S'), personalAOPC)

    return


# randomなノイズを含んだAOPCを計算する．
def randomAOPC(img_path, model_path, flag):
    seqs = sorted(glob.glob(join(img_path, "0*")))

    model = tf.keras.models.load_model(var.model_path)
    model.summary()

    # 個人の画像ディレクトリへアクセス
    for b, dir1 in enumerate(seqs):

        print("ディレクトリの現在の位置:" + dir1)

        # data generatorの作成
        train_datagen = ImageDataGenerator()

        heatmap_count = 0

        # 個人それぞれのAOPC
        personalAOPC = np.zeros((len(sorted(glob.glob(join(dir1, "0*")))), 7, 144, 2))

        # バッチごとに処理
        # ndarrayが4つ入ったリスト
        for batch in train_datagen.flow_from_personal_directory(dir1, 128, 128, 3):

            # 摂動後の画像は４つの可視化手法それぞれで摂動を行う．
            perturbationImg = copy.deepcopy(batch)

            # 摂動の回数だけ実行．今回は画像全体で行うとして画像サイズの12*12でやっている．後で変数に変更
            for i in range(12 * 12):
                # 摂動した画像を作成する関数
                perturbationImg = makePerturbationImgRandom(perturbationImg)

                # オリジナル画像と摂動画像からAOPCを算出する関数
                personalAOPC[heatmap_count:(heatmap_count + batch[0].shape[0]), :, i, :] = calculateAOPCrandom(
                    perturbationImg, batch, model)

            # print("heatmap:" + str(heatmap_count) +"\nbatchshape:" + str(batch[0].shape[0]))
            # print("shape" + str(personalAOPC[heatmap_count:(heatmap_count + batch[0].shape[0]), :, :, 0, :].shape))
            # 次のフレームの開始位置を保存
            heatmap_count = heatmap_count + batch[0].shape[0]

        now = datetime.datetime.now()
        np.save(dir1 + "/random_" + now.strftime('%Y%m%d_%H%M%S'), personalAOPC)

    return


# 可視化した結果の格納されているnpyを使いやすいように変形する関数
# heatmap ndarray frame数, 画像タイプ(3), 使用メソッド(4), 画像サイズx(12), 画像サイズy(12)
# reshape heatmap frame数, 画像タイプ(3), 使用メソッド(4), 画像サイズ(144), ランク付け及び座標(3)heatmapvalue, x, y
def ReshapeNdarray(dir1):
    start = time.time()

    # ndarrayで格納されているheatmap
    # frame数，画像のタイプ，使用するメソッド，画像サイズｘ，画像サイズｙ
    heatmap = np.load(dir1 + "/grad_cam_heatmap.npy")

    # heatmapの改造
    # frame数，画像のタイプ，使用するメソッド，ピクセル数，ヒートマップバリュー＿x＿yに変更する．

    # gridの作成[12,12]
    xx, yy = np.meshgrid(np.arange(heatmap.shape[4]), np.arange(heatmap.shape[3]))

    aaf = xx.ravel()
    bbf = yy.ravel()

    # gridの形の変換[x,3,4,144,1]
    xx = np.zeros(
        [heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3] * heatmap.shape[4]]) + xx.ravel()
    xx = np.expand_dims(xx, axis=-1)
    yy = np.zeros(
        [heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3] * heatmap.shape[4]]) + yy.ravel()
    yy = np.expand_dims(yy, axis=-1)

    # heatmapの形の変換[x,3,4,144,1]
    heatmap = np.reshape(heatmap, [heatmap.shape[0], heatmap.shape[1], heatmap.shape[2],
                                   heatmap.shape[3] * heatmap.shape[4], 1])

    # x軸y軸の結合[x,3,4,144,3]
    heatmap = np.concatenate([heatmap, yy, xx], axis=-1)

    # print("aaa")
    # heatmapの後ろをソート[x,3,4,144,3](144*3の部分はランク付けされている)
    heatmap_sorted = np.take_along_axis(heatmap, np.argsort(-heatmap[:, :, :, :, 0])[:, :, :, :, None],
                                        axis=3)

    process_time = time.time() - start
    # print("reshape:" + str(process_time))
    return heatmap_sorted


# オリジナルの画像バッチと摂動画像バッチからAOPCを計算
def calculateAOPCrandom(perturbationImg, batch, model, predictValues):
    start = time.time()

    aopc = np.zeros((batch[0].shape[0], 7, 2))

    # どの画像が推定に寄与しているかを推定するため，どの画像を摂動画像にするかを決定
    for p in range(7):
        # 顔画像だけ摂動画像
        if p == 0:
            start2 = time.time()
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [batch[0], batch[1], perturbationImg[2], batch[3]]))
            # print("aopc in loop:" + str(time.time() - start2))
        elif p == 1:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [batch[0], perturbationImg[1], batch[2], batch[3]]))
        elif p == 2:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [batch[0], perturbationImg[1], perturbationImg[2], batch[3]]))
        elif p == 3:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [perturbationImg[0], batch[1], batch[2], batch[3]]))
        elif p == 4:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [perturbationImg[0], batch[1], perturbationImg[2], batch[3]]))
        elif p == 5:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [perturbationImg[0], perturbationImg[1], batch[2], batch[3]]))
        elif p == 6:
            aopc[:, p, :] = tf.abs(predictValues - model.predict_on_batch(
                [perturbationImg[0], perturbationImg[1], perturbationImg[2], batch[3]]))

    process_time = time.time() - start
    # print("aopc:" + str(process_time))
    return aopc


# オリジナルの画像バッチと摂動画像バッチからAOPCを計算
def calculateAOPC(perturbationImg, batch, model, predictValues):
    start = time.time()

    aopc = np.zeros((batch[0].shape[0], 4, 7, 2))

    # どの可視化手法を用いるか決定
    for v in range(4):
        # どの画像が推定に寄与しているかを推定するため，どの画像を摂動画像にするかを決定
        for p in range(7):
            # 顔画像だけ摂動画像
            if p == 0:
                start2 = time.time()
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [batch[0], batch[1], perturbationImg[v][2], batch[3]]))
                # print("aopc in loop:" + str(time.time() - start2))
            elif p == 1:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [batch[0], perturbationImg[v][1], batch[2], batch[3]]))
            elif p == 2:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [batch[0], perturbationImg[v][1], perturbationImg[v][2], batch[3]]))
            elif p == 3:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [perturbationImg[v][0], batch[1], batch[2], batch[3]]))
            elif p == 4:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [perturbationImg[v][0], batch[1], perturbationImg[v][2], batch[3]]))
            elif p == 5:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [perturbationImg[v][0], perturbationImg[v][1], batch[2], batch[3]]))
            elif p == 6:
                aopc[:, v, p, :] = tf.abs(predictValues - model.predict_on_batch(
                    [perturbationImg[v][0], perturbationImg[v][1], perturbationImg[v][2], batch[3]]))

    process_time = time.time() - start
    # print("aopc:" + str(process_time))
    return aopc


# ヒートマップの情報とバッチからバッチと同じサイズの摂動後のリストを返す関数
# heatmap[x,画像のタイプ，使用するメソッド，1,3]
# prebatch 4つのndarray 画像が格納されている
def makePerturbationImgRandom(preBatch):
    start = time.time()

    # どの範囲でperturbationをするか決定する
    perturbationSize = int(ImgSize / heatmapSize)

    batch = copy.deepcopy(preBatch)

    # 目，顔ごとに処理する
    for i, img_array in enumerate(batch):

        if i != 3:
            # ランダムに初期化するための画像
            randomImg = np.random.randint(0, 255, (img_array.shape[0], perturbationSize, perturbationSize, 3))
            randomImg = randomImg.astype('float32') / 255.
            mean = np.mean(randomImg, axis=(1, 2, 3)).reshape((-1,) + (1,) * (randomImg.ndim - 1))
            # batchサイズ分のノイズ画像
            randomImg = randomImg - mean

            random = np.random.randint(0, heatmapSize, (img_array.shape[0], 2))

            # ヒートマップの位置を示している．shape[batch,2]
            perturbationPosition = (random * perturbationSize).astype(np.int)

            for b in range(batch[i].shape[0]):
                batch[i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :] = randomImg[b, :]

    # print("batch")
    process_time = time.time() - start
    # print("perturbation:" + str(process_time))
    return batch


# ヒートマップの情報とバッチからバッチと同じサイズの摂動後のリストを返す関数
# heatmap[x,画像のタイプ，使用するメソッド，1,3]
# prebatch 4つのndarrayを持つリストが４つ繋がったリスト　画像が格納されている
def makePerturbationImg(heatmap, preBatch):
    start = time.time()

    # どの範囲でperturbationをするか決定する
    perturbationSize = int(ImgSize / heatmapSize)

    batch = copy.deepcopy(preBatch)

    # 可視化手法ごとに取り出す．x,y,x+y,custom
    for m, method_list in enumerate(batch):

        # 目，顔ごとに処理する
        for i, img_array in enumerate(method_list):

            if i != 3:
                # ランダムに初期化するための画像
                randomImg = np.random.randint(0, 255, (img_array.shape[0], perturbationSize, perturbationSize, 3))
                randomImg = randomImg.astype('float32') / 255.
                mean = np.mean(randomImg, axis=(1, 2, 3)).reshape((-1,) + (1,) * (randomImg.ndim - 1))
                # batchサイズ分のノイズ画像
                randomImg = randomImg - mean

                # ヒートマップの位置を示している．shape[batch,2]
                perturbationPosition = (heatmap[0:img_array.shape[0], i, m, 1:] * perturbationSize).astype(np.int)

                for b in range(batch[m][i].shape[0]):
                    # aaaa = batch[m][i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                    #       perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :]
                    # print(aaaa.shape)
                    # bbbb = randomImg[b, :]
                    # print(bbbb.shape)
                    batch[m][i][b, perturbationPosition[b, 0]:perturbationPosition[b, 0] + perturbationSize,
                    perturbationPosition[b, 1]:perturbationPosition[b, 1] + perturbationSize, :] = randomImg[b, :]

                # batch[m][i][:, perturbationPosition[0]:perturbationPosition[0] + perturbationSize,
                # perturbationPosition[1]:perturbationPosition[1] + perturbationSize, ] = img_array[]

                # heatmap[:, :, h, :, :]

    # print("batch")
    process_time = time.time() - start
    # print("perturbation:" + str(process_time))
    return batch
