# Detecção de Imagens Falsas com Transfer Learning (ResNet18 + CIFAKE)

Miniprojeto de IA: classificar se uma imagem é **real** ou **gerada por IA** (Stable Diffusion).

## 1. Visão geral

| Item | Escolha |
|---|---|
| Tarefa | Classificação binária de imagens (REAL x FAKE) |
| Técnica | Transfer Learning com CNN (ResNet18 pré-treinada na ImageNet) |
| Dataset | CIFAKE (Kaggle): 100 mil imagens de treino e 20 mil de teste, metade reais (CIFAR-10) e metade geradas por Stable Diffusion |
| Framework | PyTorch |
| Ambiente | Google Colab (GPU T4 gratuita) |
| Tempo de execução | cerca de 5 a 15 minutos com as configurações padrão |

## 2. Arquivos

- `deteccao_imagens_falsas.ipynb`: notebook com todo o código (download, treino, avaliação).
- `README.md`: este arquivo.
- `ROTEIRO_APRESENTACAO.md`: conteúdo teórico e divisão da apresentação.

## 3. Como reproduzir (passo a passo no Google Colab)

1. Acesse https://colab.research.google.com e faça login com uma conta Google.
2. Menu **Arquivo > Fazer upload de notebook** e selecione `deteccao_imagens_falsas.ipynb`.
3. Menu **Ambiente de execução > Alterar tipo de ambiente de execução** e escolha **GPU (T4)**. Salve.
4. Menu **Ambiente de execução > Executar tudo**.
5. Aguarde. O notebook faz sozinho:
   - instala o `kagglehub` e baixa o dataset CIFAKE (não precisa de conta no Kaggle nem de chave de API);
   - prepara os dados (redimensiona para 128x128 e normaliza);
   - treina a ResNet18 por 3 épocas;
   - mostra métricas, matriz de confusão, curvas de treino e exemplos de erros.
6. Os gráficos são salvos na pasta do Colab (painel de arquivos à esquerda): `amostras.png`, `curvas_treino.png`, `matriz_confusao.png`, `erros.png`. Baixe-os para usar nos slides.
7. O modelo treinado fica em `resnet18_cifake.pt`.

> Se o download do dataset falhar, baixe o CIFAKE manualmente em
> https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images,
> envie a pasta para o Colab e ajuste a variável `DATA_ROOT` na seção 2 do notebook para apontar para a pasta que contém `train/` e `test/`.

### Dependências

O Colab já traz `torch`, `torchvision`, `scikit-learn`, `numpy` e `matplotlib`. Só o `kagglehub` é instalado pelo notebook. Para rodar localmente: `pip install torch torchvision scikit-learn matplotlib numpy pillow kagglehub`.

## 4. Como o código funciona

1. **Configurações:** número de imagens, tamanho, épocas e taxa de aprendizado ficam juntos no começo (`N_TRAIN`, `N_TEST`, `IMG_SIZE`, `EPOCHS`, `LR`). A semente (`SEED = 42`) fixa o sorteio para facilitar a reprodução.
2. **Dados:** as imagens originais têm 32x32 pixels. Elas são ampliadas para 128x128 e normalizadas com as médias e desvios da ImageNet, que é o formato esperado pela rede pré-treinada. No treino há também espelhamento horizontal aleatório (*data augmentation*).
3. **Subconjunto:** por padrão usa 20 mil imagens de treino e 5 mil de teste, para o treino ser rápido. Coloque `None` em `N_TRAIN` e `N_TEST` para usar o dataset inteiro (melhor resultado, treino mais demorado).
4. **Modelo:** ResNet18 com pesos da ImageNet. As camadas iniciais ficam **congeladas** (já sabem detectar bordas, texturas e formas). Treinamos apenas o último bloco (`layer4`) e uma nova camada final com 2 saídas (REAL e FAKE).
5. **Treino:** função de perda *CrossEntropy*, otimizador Adam, avaliação no conjunto de teste a cada época.
6. **Avaliação:** relatório com precisão, revocação e F1 por classe, matriz de confusão, curvas de loss e acurácia, e uma grade com imagens que o modelo errou.
7. **Uso com imagem própria:** a função `predict_image(caminho)` no fim do notebook retorna a probabilidade de REAL e de FAKE.

## 5. Resultados

Resultados obtidos na execução do modelo:

| Métrica | Valor |
|---|---|
| Acurácia no teste | 95,76% |
| F1 (REAL) | 0,9571 |
| F1 (FAKE) | 0,9581 |
| Tempo de treino | ~3,3 min |

## 6. Limitações

- **Generalização:** o modelo aprende os "vestígios" do gerador usado no dataset (Stable Diffusion v1.4, imagens 32x32). Em imagens de outros geradores, outras resoluções ou fotos comprimidas, o desempenho tende a cair bastante.
- **Resolução baixa:** imagens 32x32 têm poucos detalhes, o que dificulta até para humanos.
- **Corrida armamentista:** geradores novos melhoram constantemente e os detectores precisam ser retreinados.
- **Sem explicabilidade:** o modelo não mostra *por que* classificou assim (uma extensão possível é usar Grad-CAM).

## 7. Referências

- Bird, J. J.; Lotfi, A. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access, 2024. (dataset e artigo base)
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016. (ResNet)
- Rombach, R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022. (Stable Diffusion)
- Rössler, A. et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV, 2019.
- Goodfellow, I. et al. *Generative Adversarial Nets.* NeurIPS, 2014.
- Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017.
