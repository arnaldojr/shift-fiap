# ATENÇÃO: Este desafio deve ser executado em sua máquina local
# Não execute este código no navegador ou no Google Colab


# Exemplo básico da estrutura do script (salve como feature_detector_webcam.py)

import cv2
import numpy as np
    
def desenhaContorno(qp, tp, refImg, frame):
    """
    Função que desenha o contorno ao encontrar matches entre duas imagens
    
    Parâmetros:
     - qp: keypoints da imagem de consulta convertidos para np.float32
     - tp: keypoints da imagem alvo convertidos para np.float32
     - refImg: imagem de referência (template)
     - frame: imagem onde será desenhado o contorno
    """
    # Encontrar a transformação homográfica
    H, status = cv2.findHomography(qp, tp, cv2.RANSAC, 3.0)
    if H is None:
        return
    
    h, w = refImg.shape
    
    # Definir os pontos das bordas da imagem de referência
    refBorda = np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    
    # Transformar estes pontos usando a homografia encontrada
    frameBorda = cv2.perspectiveTransform(refBorda, H)
    
    # Desenhar o polígono no frame
    cv2.polylines(frame, [np.int32(frameBorda)], True, (0,255,0), 5)

def main():
    # Carregar imagem de referência
    ref_img = cv2.imread("lab_images/urso.png", 0)
    if ref_img is None:
        print("Erro: Imagem de referência não encontrada")
        return
        
    # Inicializar detector
    detector = cv2.ORB_create(nfeatures=1000)
    
    # Extrair features da imagem de referência
    kp_ref, des_ref = detector.detectAndCompute(ref_img, None)
    
    # Inicializar matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Inicializar webcam
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o frame")
            break
            
        # Converter frame para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extrair features do frame atual
        kp_frame, des_frame = detector.detectAndCompute(gray_frame, None)
        
        # Se não houver keypoints detectados, continuar
        if des_frame is None:
            cv2.imshow("Feature Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            continue
            
        # Encontrar matches
        matches = matcher.match(des_ref, des_frame)
        
        # Ordenar por distância
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Usar apenas os melhores matches
        good_matches = matches[:30] if len(matches) >= 30 else matches
        
        # Se houver matches suficientes, tentar encontrar o objeto
        MIN_MATCH_COUNT = 10
        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extrair pontos correspondentes
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
            
            # Desenhar contorno
            desenhaContorno(src_pts, dst_pts, ref_img, frame)
            
        # Desenhar os matches no frame
        img_matches = cv2.drawMatches(ref_img, kp_ref, frame, kp_frame, 
                                     good_matches, None, flags=2)
        
        # Mostrar resultado
        cv2.imshow("Feature Detection", img_matches)
        
        # Verificar se usuário quer sair
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
