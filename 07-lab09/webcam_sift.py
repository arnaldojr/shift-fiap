#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

minKPMatch = 20
sift = cv2.SIFT_create(nfeatures=500)

refImg = cv2.imread("lab_images/admiravelmundonovo.jpg", 0)
refKP, refDesc = sift.detectAndCompute(refImg, None)

vc = cv2.VideoCapture("lab_images/admiravelmundonovo.mp4")
# vc = cv2.VideoCapture(0)

bf = cv2.BFMatcher()
min_area = 1000

fonte = cv2.FONT_HERSHEY_SIMPLEX
posicao = (10, 30)
fonte_escala = 1
cor = (255, 0, 0)
espessura = 2

while True:
    rval, frame = vc.read()
    if not rval:
        break

    frameImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameKP, frameDesc = sift.detectAndCompute(frameImg, None)

    texto = "Aguardando keypoints..."

    if frameDesc is not None and refDesc is not None:
        matches = bf.knnMatch(refDesc, frameDesc, k=2)

        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)

        if len(goodMatch) > minKPMatch:
            tp = []
            qp = []
            for m in goodMatch:
                qp.append(refKP[m.queryIdx].pt)
                tp.append(frameKP[m.trainIdx].pt)
            tp, qp = np.float32((tp, qp))

            H, status = cv2.findHomography(qp, tp, cv2.RANSAC, 3.0)

            if H is not None:
                h, w = refImg.shape
                refBorda = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
                frameBorda = cv2.perspectiveTransform(refBorda, H)
                area = cv2.contourArea(frameBorda)

                if area > min_area:
                    cv2.polylines(frame, [np.int32(frameBorda)], True, (0, 255, 0), 5)
                    texto = f"Encontrado match - {len(goodMatch)}/{minKPMatch} - area: {int(area)} - BOM"
                else:
                    texto = f"Encontrado match - {len(goodMatch)}/{minKPMatch} - area: {int(area)} - INSUFICIENTE"
            else:
                texto = "Homografia inválida"
        else:
            texto = f"Encontrado match - {len(goodMatch)}/{minKPMatch} - RUIM"
    else:
        texto = "Sem descritores válidos"

    cv2.putText(frame, texto, posicao, fonte, fonte_escala, cor, espessura)
    cv2.imshow("resultado", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
