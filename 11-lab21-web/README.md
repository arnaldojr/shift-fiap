# Aula 22 - Visão Computacional com TensorFlow.js

## Objetivo

Apresentar TensorFlow.js, mostrando como implementar soluções de visão computacional no navegador.

### 1. Classificação de Imagens
- Carregamento de modelos pré-treinados
- Preprocessamento de imagens no navegador
- Interpretação de resultados

### 2. Detecção de Objetos em Tempo Real
- Acesso à webcam via JavaScript
- Modelos de detecção (COCO-SSD)
- Renderização de bounding boxes


## Como Executar

```bash
# Serve os arquivos (necessário por questões de CORS)

# Opção 1: Python
python3 -m http.server 8000 --bind 127.0.0.1

# Opção 2: Node.js
npx serve .

# Opção 3: PHP
php -S localhost:8000

# 3. Acesse http://localhost:8000
```

## Dependências

### CDN (já incluídas no HTML)
- TensorFlow.js: `@tensorflow/tfjs`
- MobileNet: `@tensorflow-models/mobilenet`
- COCO-SSD: `@tensorflow-models/coco-ssd`

### Para desenvolvimento local (opcional)
```bash
npm init -y
npm install @tensorflow/tfjs @tensorflow-models/mobilenet @tensorflow-models/coco-ssd
```

## Conceitos Importantes

### Diferenças Chave: Python vs JavaScript

| Aspecto | Python | JavaScript |
|---------|--------|------------|
| **Sintaxe** | `model.predict(data)` | `await model.predict(tensor)` |
| **Assíncrono** | Opcional | Obrigatório |
| **Tensores** | NumPy arrays | tf.Tensor objects |
| **Dispositivos** | CPU/GPU via CUDA | WebGL/CPU |
| **Deploy** | Servidor necessário | CDN/Static hosting |

## Recursos Adicionais

- [TensorFlow.js Docs](https://www.tensorflow.org/js)
- [TensorFlow Model Garden](https://github.com/tensorflow/tfjs-models)
- [ML5.js](https://ml5js.org/) - Wrapper amigável para TensorFlow.js
- [Teachable Machine](https://teachablemachine.withgoogle.com/) - Training visual sem código

