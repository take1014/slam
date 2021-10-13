#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import sys
import cv2
import math
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # detection
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        kps, des = self.orb.compute(img, kps)

        # matching
        good_matches = []
        dist_hist = None
        bins = 0
        if self.last is not None:
            dist_list = []
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    # DMatch$B7?%*%V%8%'%/%H$N;}$C$F$$$kB0@-(B
                    # $B!&(Bdistance$B!'FCD'NL5-=R;R4V$N5wN%(B
                    # $B!&(BtrainIdx$B!'3X=,5-=R;R!J;2>H%G!<%?!K$N5-=R;R$N%$%s%G%C%/%9(B
                    # $B!&(BqueryIdx$B!'%/%(%j5-=R;R!J8!:w%G!<%?!K$N5-=R;R$N%$%s%G%C%/%9(B
                    # $B!&(BimgIdx$B!'3X=,2hA|$N%$%s%G%C%/%9(B
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    good_matches.append((kp1, kp2, m.distance))
                    dist_list.append(m.distance)
                    #print(m.distance)
            dist = np.array(dist_list, dtype=np.int32)
            dist_hist, bins = np.histogram(dist, bins=10, range=(0, 100))

        final_matches = []
        if len(good_matches) > 7:
            print(len(good_matches))
            np_kp1 = np.array([pt[0] for pt in good_matches], dtype=np.float)
            np_kp2 = np.array([pt[1] for pt in good_matches], dtype=np.float)
            #if len(good_matches)
            model, inliers = ransac( (np_kp1, np_kp2),
                                     FundamentalMatrixTransform, min_samples=8,
                                     residual_threshold=1, max_trials=5000)

            for (pt1, pt2, dist), inlier in zip(good_matches, inliers):
                if inlier:
                    final_matches.append((pt1, pt2, dist))

        # return
        self.last = {'kps':kps, 'des':des}
        return final_matches, dist_hist, bins

def process_frame(extractor, img):
    # image resize
    img_col = img.shape[1]//4
    img_row = img.shape[0]//4
    img = cv2.resize(img, (img_col, img_row))

    # $BA02s%U%l!<%`$N%G!<%?$H%^%C%A$7$?E@$NAH(B
    matches, hist, bins = extractor.extract(img)
    if matches is None or hist is None:
        return

    #print(hist)
    #print(bins)
    # $B%+%&%s%HCM$NJ?6Q$r7W;;$9$k(B
    hist_mean = np.mean(hist[hist > 0])

    count = 0
    for pt1, pt2, dist in matches:
        # now frame matching point
        cx1, cy1 = map(lambda x: int(round(x)), pt1)
        # last frame matching point
        cx2, cy2 = map(lambda x: int(round(x)), pt2)

        # $B3,5iCM$r7W;;(B
        #bins = dist // 10
        # $B%$%s%G%C%/%97W;;(B
        idx = int(dist // 10)
        #print(dist, hist_idx)
        #print(hist)
        #print('hist_mean:{}, idx:{}'.format(hist_mean, idx))
        if hist[idx] >= hist_mean:
            count += 1
            cv2.circle(img, (cx1, cy1), 3, (147,20,255), 3, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx2, cy2), 3, (0,255,0), 1, lineType=cv2.LINE_AA)
            cv2.line(img, (cx1, cy1), (cx2, cy2), (255,0,0), 1)

    print('total ok count:{}'.format(count))
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

if __name__ == '__main__':
    #cap = cv2.VideoCapture('./data/production ID_4429804.mp4')
    cap = cv2.VideoCapture('./data/production ID_4309723.mp4')
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

