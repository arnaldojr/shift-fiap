# Transformers em Visão Computacional

## 1. Fundamentos dos Transformers

### 1.1 Origem e Motivação

Os Transformers foram introduzidos no artigo seminal "Attention Is All You Need" (Vaswani et al., 2017) como uma alternativa às arquiteturas recorrentes (RNNs/LSTMs) para processamento de linguagem. A inovação central foi o mecanismo de **self-attention**, que permite modelar dependências entre elementos de uma sequência sem considerar sua distância, superando limitações das arquiteturas anteriores.

A arquitetura original foi projetada para tarefas de processamento de linguagem natural (NLP), alcançando resultados sem precedentes em tradução automática e, posteriormente, dominou praticamente todas as tarefas de NLP através de variantes como BERT (Devlin et al., 2019) e GPT (Radford et al., 2018).

### 1.2 Mecanismos de Atenção

#### Self-Attention (Atenção Própria)

O mecanismo de self-attention é definido matematicamente como:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Onde:
- $Q$ (queries), $K$ (keys), e $V$ (values) são projeções lineares da entrada
- $d_k$ é a dimensionalidade das chaves (fator de escala para estabilizar o gradiente)

Este mecanismo permite que cada posição atenda a todas as posições na sequência, calculando uma média ponderada das representações de valores ($V$) onde os pesos são computados pela compatibilidade entre as consultas ($Q$) e as chaves ($K$).

#### Multi-Head Attention (Atenção Multi-Cabeça)

Para capturar diferentes tipos de relações, o Transformer utiliza várias "cabeças" de atenção em paralelo:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Onde $W_i^Q$, $W_i^K$, $W_i^V$, e $W^O$ são matrizes de parâmetros aprendíveis.

### 1.3 Embeddings Posicionais

Os Transformers não possuem noção inerente de ordem sequencial, algo crucial para a maioria das tarefas. Para contornar isso, são adicionados **embeddings posicionais** aos embeddings de entrada:

$$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

Onde:
- $pos$ é a posição na sequência
- $i$ é a dimensão
- $d_{model}$ é a dimensionalidade do modelo

Estas funções sinusoidais foram escolhidas porque permitem ao modelo extrapolar para sequências mais longas do que as vistas durante o treinamento.

## 2. Transformers em Visão: Da NLP para Imagens

### 2.1 Limitações das CNNs Tradicionais

Redes Neurais Convolucionais (CNNs) dominaram a visão computacional por uma década devido à sua eficiência e capacidade de capturar características locais através de filtros convolucionais. No entanto, apresentam limitações:

1. **Campo receptivo limitado**: Convoluções padrão operam em uma vizinhança local, exigindo múltiplas camadas para capturar dependências de longo alcance
2. **Ineficiência em modelar relações globais**: Necessitam de muitas camadas para relacionar regiões distantes da imagem
3. **Invariância translacional**: Embora útil em muitos casos, pode ser uma limitação quando o posicionamento absoluto é importante

### 2.2 Adaptando Transformers para Dados Visuais

A aplicação de Transformers em visão computacional apresenta desafios específicos:

1. **Complexidade quadrática**: O mecanismo de auto-atenção opera com complexidade O(n²), onde n é o número de tokens. Para imagens, isso se torna proibitivo rapidamente (uma imagem 256×256 teria 65.536 pixels/tokens)

2. **Ausência de informação estrutural**: Diferente de palavras em textos, pixels individuais têm pouco significado semântico isoladamente

Para resolver esses problemas, os Vision Transformers adotaram estas estratégias principais:

#### 2.2.1 Tokenização de Imagens através de Patches

Em vez de considerar pixels individuais como tokens, a abordagem do ViT (Dosovitskiy et al., 2020) divide a imagem em patches não sobrepostos, tipicamente de 16×16 ou 32×32 pixels. Cada patch é então tratado como um token e projetado linearmente para o espaço latente do modelo.

Para uma imagem de tamanho (H, W) com patches de tamanho (P, P), geramos $N = HW/P^2$ tokens.

#### 2.2.2 Incorporação de Informações Posicionais

Assim como no texto, a posição dos patches é crítica. Os embeddings posicionais são adicionados para fornecer informações sobre a localização espacial de cada patch, permitindo que o modelo aprenda relações espaciais.

## 3. Vision Transformer (ViT)

### 3.1 Arquitetura Detalhada

O Vision Transformer (ViT), introduzido por Dosovitskiy et al. (2020), foi o primeiro modelo a aplicar com sucesso a arquitetura Transformer pura para classificação de imagens.

A arquitetura do ViT consiste em:

1. **Embedding de Patches**: A imagem é dividida em patches fixos, que são linearizados e transformados em embeddings através de uma camada linear.

2. **Token de Classificação [CLS]**: Um token especial aprendível é incluído no início da sequência, cuja representação final é usada para classificação.

3. **Embeddings Posicionais**: Adicionados aos embeddings de patches para codificar informação posicional.

4. **Camadas do Transformer**: Os embeddings processados são passados por múltiplas camadas de Transformer, cada uma com Multi-Head Self-Attention e MLP Feed-Forward.

5. **Cabeça de Classificação**: Uma camada MLP sobre o token [CLS] final para previsão de classe.

![Vision Transformer Architecture](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

Matematicamente, para uma imagem $x \in \mathbb{R}^{H \times W \times C}$ dividida em $N$ patches, temos:

$$z_0 = [x_{class}; x_p^1E; x_p^2E; ...; x_p^NE] + E_{pos}$$

Onde:
- $x_{class}$ é o token de classificação
- $x_p^i$ é o i-ésimo patch
- $E$ é a matriz de projeção
- $E_{pos}$ são os embeddings posicionais

### 3.2 Análise de Desempenho e Escalabilidade

O ViT demonstrou que, com treinamento em conjuntos de dados grandes o suficiente (como JFT-300M), pode superar CNNs de última geração. Observações importantes:

1. **Eficiência de dados vs. indução de viés**: CNNs incorporam fortes indutores de viés (localidade, invariância à translação) que ajudam em regime de dados limitados. O ViT tem menos viés intrínseco, exigindo mais dados para generalizar bem.

2. **Escalabilidade**: O ViT escala extremamente bem com mais dados e tamanho do modelo.

3. **Transferência de conhecimento**: Modelos pré-treinados em grandes conjuntos de dados transferem eficazmente para tarefas com poucos dados.

### 3.3 Análise de Atenção e Interpretabilidade

Os mapas de atenção do ViT revelam propriedades interessantes:

1. **Atenção de baixo nível**: Nas primeiras camadas, a atenção tende a focar em patches locais próximos, imitando o comportamento de receptores de campo de CNNs.

2. **Atenção de alto nível**: Em camadas posteriores, o foco se expande para relações semânticas de longo alcance, atendendo a objetos semanticamente relevantes.

3. **Atenção baseada em características**: O modelo aprende a atender a características visuais importantes sem supervisão explícita.

## 4. Variantes e Melhorias

### 4.1 Swin Transformer

O Swin Transformer (Liu et al., 2021) introduz uma abordagem hierárquica com atenção local deslocada em janelas, abordando as limitações de escala do ViT:

1. **Atenção em Janelas**: Calcula atenção apenas dentro de janelas locais, reduzindo a complexidade de O(n²) para O(m²), onde m é o tamanho da janela.

2. **Janelas Deslocadas**: Alternam entre configurações de janelas regulares e deslocadas para permitir conexões entre janelas.

3. **Hierarquia**: Fusão progressiva de tokens para reduzir resolução e aumentar canais, criando uma representação hierárquica similar a uma CNN.

![Swin Transformer](https://user-images.githubusercontent.com/24825165/121768619-038e6d80-cb9a-11eb-8cb7-daa827e7772b.png)

### 4.2 DeiT (Data-efficient Image Transformers)

DeiT (Touvron et al., 2021) mostra que os Vision Transformers podem ser treinados eficientemente em conjuntos de dados menores (como ImageNet-1K) sem recorrer a enormes conjuntos pré-treinados:

1. **Destilação de Conhecimento**: Utiliza uma CNN como professor para transferir conhecimento para o ViT.

2. **Token de Destilação**: Adiciona um token específico para destilação além do token de classificação.

3. **Técnicas Avançadas de Regularização e Aumento de Dados**: Emprega estratégias como CutMix, Mixup e RandAugment para compensar a falta de grandes conjuntos de dados.

### 4.3 PVT (Pyramid Vision Transformer)

O Pyramid Vision Transformer (Wang et al., 2021) adapta os transformers para várias tarefas de visão, como detecção e segmentação:

1. **Estrutura piramidal**: Reduz progressivamente a resolução e aumenta a dimensionalidade dos canais, criando mapas de características multi-escala.

2. **Atenção de Redução Espacial**: Reduz a complexidade computacional ao aplicar pooling na dimensão espacial antes do cálculo de atenção.

### 4.4 MViTv2 (Multiscale Vision Transformers)

O MViTv2 (Fan et al., 2022) é uma arquitetura multi-escala otimizada:

1. **Pool-Attention**: Operação de pool nas queries e keys para reduzir complexidade.

2. **Residual Pooling Connections**: Conexões residuais entre diferentes escalas.

3. **Decomposição em Eixos Separados**: Atenção separada para dimensões espaciais.

## 5. Arquiteturas Híbridas: CNN + Transformers

### 5.1 ConviT (Convolutional Vision Transformer)

O ConViT (d'Ascoli et al., 2021) introduz tokens de gating suaves que permitem ao modelo transitar suavemente entre atenção convolucional e não-local:

1. **Inicialização de Atenção Convolucional**: Inicializa os valores de atenção para emular convolução.

2. **Tokens de Gating**: Permitem que o modelo aprenda a equilibrar entre características convolucionais e de atenção global.

### 5.2 CvT (Convolutional vision Transformer)

O CvT (Wu et al., 2021) introduz:

1. **Projeções Convolucionais**: Substitui as projeções lineares em Q, K, V por projeções convolucionais.

2. **Tokenização Convolucional Hierárquica**: Usa convoluções para criar representações em diferentes escalas.

### 5.3 MobileViT e EfficientFormer

Arquiteturas como MobileViT (Mehta & Rastegari, 2021) e EfficientFormer (Li et al., 2022) focam em eficiência para dispositivos móveis:

1. **Atenção Local**: Aplicam atenção apenas dentro de blocos espaciais locais.

2. **Combinação Estratégica**: Alternam entre convoluções (eficientes) e blocos de atenção (expressivos).

3. **Operações leves**: Utilizam convoluções separáveis em profundidade e outras otimizações de parâmetros e computação.

## 6. Aprendizado Auto-Supervisionado e Pré-Treinamento

### 6.1 DINO (Self-Distillation with No Labels)

DINO (Caron et al., 2021) é um método de aprendizado auto-supervisionado baseado em destilação:

1. **Arquitetura de Estudante-Professor**: Duas redes com a mesma arquitetura, mas diferentes parâmetros.

2. **Aumento de Dados Contrastivo**: Visões diferentes da mesma imagem são processadas pelo estudante e professor.

3. **Destilação**: O estudante é treinado para prever as distribuições de saída do professor.

4. **Momento no Professor**: Os parâmetros do professor são atualizados como média móvel dos parâmetros do estudante.

DINO produz mapas de atenção que segmentam objetos sem supervisão e aprende características úteis para várias tarefas downstream.

### 6.2 MAE (Masked Autoencoders)

MAE (He et al., 2022) é inspirado no sucesso do BERT em NLP:

1. **Mascaramento Aleatório de Patches**: Oculta aleatoriamente grande parte da imagem (75%).

2. **Codificador-Decodificador Assimétrico**: Apenas patches visíveis passam pelo codificador Transformer, economizando computação.

3. **Reconstrução**: O decodificador leve reconstrói os patches mascarados.

![Masked Autoencoder](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9Jec3nnOGbWrQ5_SjPf2Ow.png)

### 6.3 BEiT (Bidirectional Encoder representation from Image Transformers)

BEiT (Bao et al., 2021) segue uma abordagem de modelagem de linguagem visual:

1. **Tokenização Visual**: Usa um VQ-VAE para transformar a imagem em tokens visuais discretos.

2. **Previsão de Tokens Mascarados**: Treina o modelo para prever os tokens visuais dos patches mascarados.

## 7. Aplicações Específicas

### 7.1 Detecção de Objetos com DETR

DETR (Carion et al., 2020) reformula a detecção de objetos como um problema de previsão direta de conjunto:

1. **Encoder-Decoder Transformer**: O encoder processa features da CNN, e o decoder gera N previsões em paralelo.

2. **Queries de Objeto**: O decoder utiliza N queries de objeto aprendíveis para identificar objetos.

3. **Bipartite Matching**: Pareamento ótimo entre previsões e ground truth para calcular perda.

4. **End-to-End**: Elimina a necessidade de âncoras e algoritmos de supressão de não-máximos.

![DETR Architecture](https://lh5.googleusercontent.com/rPXh8HHA-KnqdFuTjXsqBLS3A1y1QGZ5eQnqt3RFV-ZcogCZlku7VpnFFeIcIBmIYN4nevzDlWOEq69nS2d_ZDWM5fO_RgZ7OJrAh6X5h3OvLLXlqBDc3vGvTvfQRTG2KAu7Mv3tH4QJ4Nl_OtSHxdo)

### 7.2 Segmentação com Mask2Former

Mask2Former (Cheng et al., 2021) unifica instância, semântica e segmentação panóptica:

1. **Queries de Máscara**: Queries específicas que aprendem diretamente a prever máscaras.

2. **Atenção Entre Masking Tokens**: Permite que o modelo foque nas regiões relevantes da imagem.

3. **Predição Unificada**: Um único modelo treina e infere diferentes tipos de segmentação.

### 7.3 CLIP (Contrastive Language-Image Pre-training)

CLIP (Radford et al., 2021) conecta texto e imagens:

1. **Treinamento Contrastivo**: Aprende a associar textos a imagens correspondentes.

2. **Zero-shot Transfer**: Permite classificar imagens em novas categorias sem treinamento adicional.

3. **Robustez**: Demonstra surpreendente generalização para distribuições fora do treinamento.

## 8. Estado Atual e Futuro dos Vision Transformers

### 8.1 Avanços Recentes

1. **Eficiência Computacional**: Modelos como MobileViT e TinyViT adaptam transformers para computação limitada.

2. **Combinação de Modalidades**: Modelos como CLIP e Flamingo integram visão com texto e outras modalidades.

3. **Geração de Imagens**: Modelos difusivos com componentes transformer estão redefinindo a geração de imagens (DALL-E, Stable Diffusion).
