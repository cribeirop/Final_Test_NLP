# Sumarização de Contos do Dataset Tiny Stories

## Visão Geral

Neste projeto, realizou-se um processo de sumarização de contos utilizando uma porção do dataset [Tiny Stories](https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification). O roteiro deste projeto está disponível com o nome `Roteiro.pdf`.

A combinação das técnicas utilizadas provou ser valiosa para gerar resumos com desempenho excelente nas métricas ROUGE. Esse projeto valida a aplicabilidade de modelos como o BERT e BART em tarefas de sumarização extrativa e abstrativa, e oferece perspectivas sobre como diferentes técnicas podem ser combinadas para aprimorar a qualidade dos resumos gerados.

## Instalação

Siga os passos abaixo para instalar e configurar o ambiente do projeto:

### Pré-requisitos

- **Python**: 3.8 ou superior
- **Pip**: Certifique-se de ter o gerenciador de pacotes `pip` instalado.

### Passo 1: Clone o repositório

Clone o repositório para sua máquina local usando o seguinte comando:

```bash
git clone https://github.com/cribeirop/TinyStoriesSummarization
```
```bash
cd .\TinyStoriesSummarization\
```

### Passo 2: Crie um ambiente virtual (opcional, mas recomendado)
Crie um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv venv
```
```bash
source venv/bin/activate  # No Linux ou macOS
venv\Scripts\activate     # No Windows
```

```bash
pip install -r requirements.txt
```

### Passo 3: Rode o arquivo main.py para acompanhar a implementação do projeto

```bash
python3 main.py
```