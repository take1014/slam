#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import sys
import cv2
import math
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last = None

    def extract(self, img):
        # get features
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=3)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        kps, des = self.orb.compute(img, kps)

        # matching
        matches = None
        if self.last is not None:
            print(self.last['des'])
            matches = self.bf.match(des, self.last['des'])
            # DMatch型オブジェクトの持っている属性
            # ・distance：特徴量記述子間の距離
            # ・trainIdx：学習記述子（参照データ）の記述子のインデックス
            # ・queryIdx：クエリ記述子（検索データ）の記述子のインデックス
            # ・imgIdx：学習画像のインデックス
            matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches])

        #matches = self.bf.match(des, self.last['des'])
        self.last = {'kps':kps, 'des':des}

        return matches
        #return [kps[m.queryIdx] for m in matches], [kps[m.trainIdx] for m in matches]
        #return kps, des

def process_frame(extractor, img):
    # image resize
    img_col = img.shape[1]//4
    img_row = img.shape[0]//4
    img = cv2.resize(img, (img_col, img_row))

    # 前回フレームのデータとマッチした点の組
    matches = extractor.extract(img)

    if matches is None:
        return

    for pt1, pt2 in matches:
        print(pt1.pt)
        print(pt2.pt)

        # now frame matching point
        cx1, cy1 = map(lambda x: int(round(x)), pt1.pt)
        # last frame matching point
        cx2, cy2 = map(lambda x: int(round(x)), pt2.pt)

        cv2.circle(img, (cx1, cy1), 3, (0,0,255), 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx2, cy2), 3, (0,255,0), 1, lineType=cv2.LINE_AA)

        norm = math.sqrt((pt1.pt[0]-pt2.pt[0])**2 + (pt1.pt[1]-pt2.pt[1])**2)
        print(norm)
        if norm < 30:
            cv2.line(img, (cx1, cy1), (cx2, cy2), (255,0,0), 1)

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

if __name__ == '__main__':
    cap = cv2.VideoCapture('./data/production ID_4429804.mp4')
    # create detector
    extractor = FeatureExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(extractor, frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

