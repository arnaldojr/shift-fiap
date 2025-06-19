import cv2
import numpy as np
import random
import time

# ==========================
# Constantes
# ==========================
TAMANHO_BLOCO = 80
INTERVALO_SPAWN = 1  # segundos entre novos blocos

# Cores em BGR
VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AZUL = (255, 0, 0)
AMARELO = (0, 255, 255)
ROXO = (128, 0, 128)
CORES = [VERDE, VERMELHO, AZUL, AMARELO, ROXO]

# Configuração da detecção da luva verde
LOWER_VERDE = np.array([35, 70, 70])  # Limite inferior HSV para verde
UPPER_VERDE = np.array([85, 255, 255])  # Limite superior HSV para verde

# ==========================
# Inicialização da webcam
# ==========================
cap = cv2.VideoCapture(1)  # Ajuste para 0 se necessário
if not cap.isOpened():
    raise IOError("Erro ao acessar a câmera")

LARGURA_JANELA = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ALTURA_JANELA = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Resolução detectada:", LARGURA_JANELA, "x", ALTURA_JANELA)

# ==========================
# Classe Bloco
# ==========================
class BlocoCaindo:
    def __init__(self):
        self.largura = TAMANHO_BLOCO
        self.altura = TAMANHO_BLOCO
        self.x = random.randint(0, LARGURA_JANELA - self.largura)
        self.y = 0
        self.velocidade = 5
        self.cor = random.choice(CORES)
        self.ativo = True

    def atualizar(self):
        self.y += self.velocidade
        if self.y + self.altura > ALTURA_JANELA:
            self.ativo = False

    def desenhar(self, frame):
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.largura, self.y + self.altura),
                      self.cor, -1)

    def verificar_colisao(self, ponto_x, ponto_y):
        return (self.x <= ponto_x <= self.x + self.largura and
                self.y <= ponto_y <= self.y + self.altura)

# ==========================
# Função de detecção da luva
# ==========================
def detectar_luva_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_VERDE, UPPER_VERDE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) > 500:
            x, y, w, h = cv2.boundingRect(maior)
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            return cx, cy, True
    return 0, 0, False

# ==========================
# Função principal do jogo
# ==========================
def jogo_tetris_cv():
    blocos = []
    ultimo_tempo_spawn = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        frame = cv2.flip(frame, 1)  # espelha horizontalmente

        # Detecta a luva verde
        cx_luva, cy_luva, luva_detectada = detectar_luva_verde(frame)

        # Gera novo bloco se passou o tempo
        if time.time() - ultimo_tempo_spawn > INTERVALO_SPAWN:
            blocos.append(BlocoCaindo())
            ultimo_tempo_spawn = time.time()

        # Atualiza blocos e verifica colisões
        blocos_ativos = []
        for bloco in blocos:
            bloco.atualizar()
            if bloco.ativo:
                bloco.desenhar(frame)
                if luva_detectada and bloco.verificar_colisao(cx_luva, cy_luva):
                    texto = "Pegou o bloco!"
                    cv2.putText(frame, texto, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue  # ignora este bloco
                blocos_ativos.append(bloco)

        blocos = blocos_ativos

        texto = "Luva verde detectada" if luva_detectada else "Luva verde nao detectada"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Jogo CV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Execução
# ==========================
if __name__ == "__main__":
    jogo_tetris_cv()
