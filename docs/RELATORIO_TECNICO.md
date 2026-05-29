# RELATÓRIO TÉCNICO: PARALLEL FACE SEARCH
**Disciplina:** Sistemas Paralelos e Distribuídos  
**Professor:** [Nome do Professor]  
**Grupo:** [Integrantes do Grupo]  

---

## 1. Título do Projeto
**Parallel Face Search**: Reconhecimento Facial em Tempo Real com Busca Paralela em Múltiplos Processos e Resiliência a Máscaras.

---

## 2. Descrição do Problema (Etapa 1)
O problema a ser resolvido é a **identificação de indivíduos em tempo real** (a partir de uma webcam ou fluxo de vídeo) contra uma galeria pública de pessoas procuradas (ex: dados públicos do Ministério da Justiça e Segurança Pública - MJSP).

Este problema apresenta dois desafios centrais:
1. **CPU-Bound no Matching Facial**: O cálculo de similaridade de cosseno entre o vetor da face capturada (512 dimensões) e a base inteira de procurados (que pode conter milhares de registros) precisa rodar a cada frame. De forma sequencial, essa busca gera latência, derrubando o frame rate da câmera.
2. **Oclusão Facial (Máscaras/Acessórios)**: Em cenários reais de segurança pública, suspeitos podem cobrir partes do rosto com máscaras, toucas ou balaclavas. Os algoritmos tradicionais de reconhecimento facial falham porque dependem do rosto completo.

A relevância do projeto está em demonstrar como técnicas de **Sistemas Paralelos** conseguem viabilizar o processamento biométrico em tempo real em hardware comum, enquanto técnicas de visão computacional (fatiamento de embedding) mantêm o sistema funcional sob oclusões faciais severas.

---

## 3. Justificativa do Paralelismo (Etapa 1)
A codificação da face de entrada em vetores 512d (inferência Deep Learning com ArcFace) é acelerada via GPU ou otimizações nativas de CPU (ONNX Runtime). No entanto, a **comparação de similaridade um-para-todos** contra a galeria é puramente matemática (cosseno de vetores). 

Ao realizar a comparação de forma sequencial:
- Um único núcleo do processador realiza sucessivos loops de multiplicação vetorial.
- O tempo total cresce linearmente ($O(N)$) com o tamanho da galeria.
- Para múltiplos rostos em cena, o tempo de processamento supera o intervalo de 33ms por frame necessário para manter 30 FPS estáveis na webcam.

**Benefício da abordagem paralela (Master-Worker)**:
Ao dividir a galeria em fatias e distribuí-la para múltiplos processos em paralelo, o tempo de busca teórica cai para $T_p \approx T_s / P$, onde $P$ é o número de núcleos de processador ativos. O paralelismo converte o tempo de resposta em latência aceitável, otimizando o throughput (faces analisadas por segundo) e a experiência de uso.

---

## 4. Arquitetura da Solução (Etapa 2)

O sistema implementa um modelo **Híbrido de Programação Concorrente**:
1. **Produtor-Consumidor (Multithreading)** para interface e captura de imagem.
2. **Master-Worker (Multiprocessing)** para o cálculo paralelo de similaridade.

### Diagrama Arquitetural de Fluxo
```mermaid
graph TD
    subgraph Thread Principal (Interface e Captura)
        Webcam[Captura OpenCV] -->|Frame BGR| SharedFrame[Buffer Compartilhado]
        SharedFrame -->|Exibição HUD| Screen(Janela OpenCV - imshow)
    end

    subgraph Thread de Reconhecimento (Consumidora/Master)
        SharedFrame -->|Lê frame recente| Detection[Detector SCRFD]
        Detection -->|Alinhamento & Recorte| Embedder[Codificador ArcFace 512d]
        Embedder -->|3 embeddings por rosto| SimilarityMaster[Similarity Search Master]
    end

    subgraph Workers de Comparação (Multiprocessing)
        SimilarityMaster -->|Envia vetor| W1[Processo Worker 1]
        SimilarityMaster -->|Envia vetor| W2[Processo Worker 2]
        SimilarityMaster -->|Envia vetor| W3[Processo Worker N]
        
        W1 -->|Fatia 1 da Galeria| Similarity1[Cosine Similarity]
        W2 -->|Fatia 2 da Galeria| Similarity2[Cosine Similarity]
        W3 -->|Fatia N da Galeria| SimilarityN[Cosine Similarity]
        
        Similarity1 -->|Melhores Matches| SimilarityMaster
        Similarity2 -->|Melhores Matches| SimilarityMaster
        SimilarityN -->|Melhores Matches| SimilarityMaster
    end

    SimilarityMaster -->|Combina e ordena| SharedFrame
```

### Componentes e Comunicação
- **Componentes do Sistema**:
  - `webcam/pipeline.py`: Orquestrador principal. Garante que a webcam capture a 30 FPS na thread principal e envia frames para a thread de processamento por meio de variáveis de estado com travas (`threading.Lock()`).
  - `engine/`: Extrai o rosto completo, a metade superior (máscara) e a região ocular. Gera 3 embeddings distintos por face detectada.
  - `matching/search.py`: Gerencia a pool distribuída. Divide os candidatos da base usando algoritmo round-robin para garantir balanceamento de carga entre os processos.
- **Divisão de Tarefas**: O trabalho de comparação vetorial é particionado. Cada worker possui uma variável global persistente contendo sua fração dos candidatos (galeria fatiada). O Master envia apenas os 512 floats do rosto a ser pesquisado, minimizando o custo de serialização (IPC) de dados pesados.
- **Ferramentas utilizadas**: Python (multiprocessing, concurrent.futures, threading, OpenCV, numpy, insightface).

---

## 5. Tecnologias Utilizadas (Etapa 2)
- **InsightFace (SCRFD & ArcFace)**: Modelos de Deep Learning para detecção de rostos ultrarrápida (SCRFD) e extração de representações faciais robustas (ArcFace).
- **OpenCV**: Captura e manipulação de vídeo e renderização de elementos de HUD na tela.
- **Cloudflare R2**: Armazenamento em nuvem compatível com S3 para armazenar de forma distribuída o manifesto e vetores de faces da base oficial.
- **ProcessPoolExecutor (Python Stdlib)**: Criação e controle do ciclo de vida dos processos workers.
- **psutil**: Captura nativa do sistema operacional (Windows/Linux) de consumo de memória RAM física (RSS) e porcentagem de uso de CPU.

---

## 6. Explicação da Implementação (Etapa 3)

O código segue padrões limpos de engenharia de software estruturado em módulos independentes:
- **Tratamento de Erros**: O sistema valida e trata quedas de conexão com o Cloudflare R2 utilizando um manifesto local cacheado quando necessário. Tratamento de exceções na webcam evita falhas críticas caso a câmera seja desconectada em tempo de execução.
- **Instruções de Instalação e Execução**: Detalhadas de forma concisa em linha de comando no repositório.
- **Uso do Paralelismo**: Implementado de forma clara e configurável através do parâmetro `--mode parallel` ou `--mode benchmark` e o argumento `--workers N`.

---

## 7. Testes e Análise de Desempenho (Etapa 4)

### Metodologia de Testes
Para atender à Etapa 4, o sistema coleta os dados utilizando um script autônomo (`face-search-benchmark`) que avalia o algoritmo simulando buscas contra bases de diferentes tamanhos e alternando a contagem de núcleos (workers) alocados para o processamento.

As métricas coletadas incluem:
- **Tempo Total**: Duração total da busca na base.
- **Latência**: Tempo médio (ms) para concluir uma única busca.
- **Speedup**: Razão entre o tempo sequencial e o tempo paralelo ($S = T_s / T_p$).
- **Eficiência**: Speedup normalizado pelo número de workers ($E = S / N_{workers}$).
- **Throughput**: Número de requisições resolvidas por segundo.
- **Uso de CPU e Memória**: Percentuais de CPU consumidos e RAM alocada.

### Resultados Obtidos (Espaço para Preenchimento)

*Nota: Os dados abaixo são de referência teórica/simulação inicial. Devem ser atualizados após a execução do script de benchmark final no hardware da máquina local.*

#### Tabela 1: Escalonamento de Processos (Galeria Fixa em 1.000 Registros, repetições = 100)
| Cenário | Workers | Latência Média (ms) | Speedup | Eficiência | Throughput (Faces/s) | Uso CPU (%) | Uso RAM (MB) |
|---|---|---|---|---|---|---|---|
| Sequencial | 1 (Seq) | `[Inserir]` | 1.00x | 100.0% | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| Paralelo | 2 | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| Paralelo | 4 | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| Paralelo | 8 | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |

#### Tabela 2: Impacto do Tamanho da Galeria (Usando 4 Workers, repetições = 100)
| Tamanho da Galeria | Latência Seq (ms) | Latência Par (ms) | Speedup Obtido | Eficiência |
|---|---|---|---|---|
| 100 Registros | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| 1.000 Registros | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| 5.000 Registros | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |
| 10.000 Registros | `[Inserir]` | `[Inserir]` | `[Inserir]` | `[Inserir]` |

---

## 8. Análise Crítica dos Resultados (Etapa 4)
*Orientação para análise acadêmica a ser redigida baseada nas medições coletadas:*
- **Análise do Overhead**: Quando a galeria é pequena (ex: 100 registros), o overhead de IPC (Inter-Process Communication) para enviar as requisições e serializar as respostas dos subprocessos consome tempo considerável. Nesses casos, o Speedup pode ser menor que 1.0x (sequencial roda mais rápido).
- **Escalabilidade**: Conforme o tamanho da base cresce (ex: 5.000 registros ou mais), a computação pura de similaridade começa a dominar sobre o overhead de comunicação. O ganho de velocidade (Speedup) se aproxima do número físico de cores do processador, provando a viabilidade da arquitetura paralela para cenários corporativos/de larga escala.
- **Uso de Recursos**: O monitoramento mostra que o consumo de memória RAM aumenta de forma multiplicativa conforme mais workers são criados, pois cada worker mantém sua fatia da galeria em seu espaço de memória privada. Há um tradeoff clássico de computação paralela: troca-se uso de memória RAM por velocidade de processamento.

---

## 9. Conclusão e Melhorias Futuras (Etapa 5)
O projeto demonstrou com sucesso a aplicação de computação multiprocessada para resolver o gargalo de matching facial em tempo real. A separação do motor de processamento em threads garantiu que a interface visual OpenCV permanecesse fluida (30 FPS na captura), mesmo quando a carga do benchmark aumentava consideravelmente.

**Melhorias identificadas para trabalhos futuros**:
1. **Compartilhamento de Memória Nativo**: Utilizar um bloco de memória compartilhada nativo (ex: `multiprocessing.shared_memory`) ou bancos vetoriais otimizados na RAM para evitar duplicação física da galeria nos workers.
2. **GPU Acceleration**: Utilizar CUDA para paralela de dados massivos na placa gráfica (onde o cálculo de cosseno pode ser paralelizado em milhares de threads da GPU).
3. **Distribuição em Rede**: Expandir a arquitetura mestre-trabalhador local para nós em rede usando gRPC ou REST APIs, movendo o processador da galeria de um ambiente single-host para um cluster de servidores distribuídos.
