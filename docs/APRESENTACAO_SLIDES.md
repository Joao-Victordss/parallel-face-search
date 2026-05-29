# ESTRUTURA DOS SLIDES DE APRESENTAÇÃO
**Projeto:** Parallel Face Search  
**Disciplina:** Sistemas Paralelos e Distribuídos  

Este documento serve como roteiro para a montagem dos slides (no PowerPoint, Google Slides, Marp ou similar) e fornece o roteiro de falas sugerido para a apresentação em sala de aula (Etapa 5).

---

## Slide 1: Capa do Projeto
*   **Título:** Parallel Face Search
*   **Subtítulo:** Busca Facial Paralela Multiprocessada Resiliente a Oclusões e Máscaras
*   **Apresentadores:** [Inserir Nomes]
*   **Curso/Disciplina:** Sistemas Paralelos e Distribuídos
*   **Identidade Visual Sugerida:** Fundo escuro (Dark Mode), tons de azul neon e ciano, imagem estilizada de malha de identificação facial ou ícone de CPU/Webcam.

---

## Slide 2: O Problema (Etapa 1)
*   **Conteúdo Visual:**
    *   Imagens de rostos usando máscaras, toucas ou balaclavas.
    *   Um fluxograma simples mostrando o gargalo: `Captura de Frame (30ms)` -> `Reconhecimento (15ms)` -> `Busca na base (50ms - Gargalo)` -> `Total = 95ms (Reduz FPS para ~10)`.
*   **Tópicos Chave:**
    *   **Identificação em Tempo Real**: Fazer o matching rápido de faces detectadas na webcam contra uma galeria de procurados.
    *   **Gargalo de CPU**: O cálculo de similaridade de cosseno em bancos vetoriais volumosos consome ciclos de CPU cruciais.
    *   **Oclusão Facial**: Modelos tradicionais de face falham quando o indivíduo usa máscara.
*   **Roteiro de Fala:**
    > "Olá a todos. Nosso projeto resolve o problema do reconhecimento de indivíduos em tempo real sob cenários realistas de segurança pública, onde as pessoas podem estar usando máscaras ou bonés. Esse cenário traz dois desafios: o algoritmo precisa ser rápido para manter o frame rate fluido e precisa reconhecer o suspeito mesmo se ele estiver mascarado. O gargalo computacional está no loop matemático de comparação das faces contra o banco de dados."

---

## Slide 3: Justificativa do Paralelismo (Etapa 1)
*   **Conteúdo Visual:**
    *   Gráfico comparativo conceitual entre a busca sequencial ($O(N)$) e a busca paralela ($O(N/P)$).
*   **Tópicos Chave:**
    *   Busca sequencial executa loops repetitivos em um único núcleo de processador.
    *   A inferência de IA (ArcFace) é acelerada, mas a busca por similaridade é CPU-bound.
    *   Paralelismo de dados fatiando a galeria nos múltiplos núcleos da CPU.
*   **Roteiro de Fala:**
    > "Por que paralelismo? A extração das características do rosto gera um vetor de 512 números. Comparar esse vetor com o banco de procurados de forma sequencial é lento. Em uma base de dados real, se fizermos isso um a um, o sistema vai travar o vídeo da câmera. Paralelizando essa busca, nós dividimos a base de procurados entre os núcleos da CPU, processando várias partes simultaneamente e mantendo o vídeo liso a 30 frames por segundo."

---

## Slide 4: Arquitetura da Solução (Etapa 2)
*   **Conteúdo Visual:**
    *   Diagrama em blocos da arquitetura Master-Worker.
    *   Destaque para o fatiamento da galeria na inicialização do pool.
*   **Tópicos Chave:**
    *   **Produtor-Consumidor**: Threads separadas para captura da câmera e orquestração de reconhecimento.
    *   **Master-Worker**: Processo pai (Master) delega a busca para subprocessos filhos (Workers) via `ProcessPoolExecutor`.
    *   **Minimização de IPC**: Workers mantêm a base persistente na memória para evitar tráfego de dados volumosos entre processos a cada frame.
*   **Roteiro de Fala:**
    > "Nossa arquitetura usa duas técnicas concorrentes. Uma thread no processo pai cuida exclusivamente de ler a webcam e atualizar a tela, para o vídeo nunca travar. Outra thread atua como Master de Reconhecimento. Quando um rosto é detectado, ela envia apenas o vetor de busca para múltiplos processos Workers locais. Cada worker já tem em sua própria memória RAM uma fatia do banco de procurados (distribuída em round-robin). Eles fazem a busca local de cosseno em paralelo e devolvem só os melhores candidatos ao Master, economizando tempo de comunicação entre processos."

---

## Slide 5: Tratamento de Máscaras (A Diferencial da Solução)
*   **Conteúdo Visual:**
    *   Tabela dos três caminhos de comparação (Face Exposta, Face Coberta, Olhos).
    *   Imagem ilustrando o corte das 3 regiões da face.
*   **Tópicos Chave:**
    *   Três variantes de vetores gerados por suspeito (`full`, `upper`, `periocular`).
    *   Comparação paralela tripla executada de forma simultânea.
    *   Acúmulo temporal de evidências entre frames para tomada de decisão robusta.
*   **Roteiro de Fala:**
    > "Para vencer as máscaras, em vez de classificar se a pessoa está mascarada ou não, nós dividimos a face de entrada em três regiões: o rosto completo, a metade superior e a região do entorno dos olhos. A galeria de procurados também armazena esses três vetores. O sistema compara as três regiões em paralelo. Se o suspeito estiver de máscara, a similaridade da parte inferior vai falhar, mas a região dos olhos ou a metade superior vai casar, acumulando evidência nos frames consecutivos até termos certeza da identidade."

---

## Slide 6: Metodologia de Testes e Desempenho (Etapa 4)
*   **Conteúdo Visual:**
    *   Lista das métricas medidas (Tempo Total, Speedup, Eficiência, Vazão, CPU e RAM).
    *   Foto/Print do console exibindo o relatório do benchmark final.
*   **Tópicos Chave:**
    *   Uso de `psutil` para auditoria fina de hardware durante os testes.
    *   Validação de desempenho sob diferentes contagens de Workers (1, 2, 4, 8, etc.).
    *   Benchmark contra diferentes tamanhos de entrada (Galeria com centenas a milhares de registros).
*   **Roteiro de Fala:**
    > "Para testar o desempenho do sistema e cumprir a Etapa 4, nós implementamos um módulo de medição nativo usando a biblioteca `psutil`. Nós testamos o comportamento do sistema sob diversas configurações de trabalhadores, desde o modo puramente sequencial de 1 worker até o limite de núcleos físicos do processador. Coletamos métricas de Speedup, Eficiência no uso de CPU, pico de consumo de RAM e Throughput de faces processadas."

---

## Slide 7: Resultados Obtidos (Etapa 4)
*   **Conteúdo Visual:**
    *   *Tabelas ou gráficos de Speedup e Eficiência gerados a partir do benchmark real.*
*   **Tópicos Chave:**
    *   Apresentação dos tempos sequencial vs. paralelo.
    *   Ponto de equilíbrio de overhead de IPC (Tradeoff de comunicação vs. processamento).
    *   Discussão do consumo de memória de workers adicionais.
*   **Roteiro de Fala:**
    > "Aqui estão nossos resultados práticos. Conforme podemos ver na tabela, o speedup cresce substancialmente à medida que a galeria aumenta. Em bases muito pequenas, o tempo de comunicação via pipe consome parte da vantagem do paralelismo. Porém, com galerias maiores, a busca paralela se destaca de forma exponencial, atingindo uma eficiência de [Inserir %] com [Inserir N] workers e permitindo que o sistema funcione com baixíssima latência."

---

## Slide 8: Dificuldades e Melhorias (Etapa 5)
*   **Tópicos Chave:**
    *   **Dificuldades**: Custo de serialização do Python no Windows (que usa `spawn` em vez de `fork`), gerando mais latência inicial para subir os workers; consumo de RAM duplicada por worker.
    *   **Soluções aplicadas**: Carregamento persistente global nos workers na inicialização e fatiamento round-robin para balancear a base.
    *   **Evolução**: Utilização de memória compartilhada nativa (`SharedMemory`) ou banco de vetores nativo (ex: FAISS).
*   **Roteiro de Fala:**
    > "Entre as principais dificuldades encontradas, destacamos o comportamento do Windows que não permite fazer 'fork' direto de processos na memória, exigindo criar subprocessos limpos que demoram alguns segundos a mais para iniciar e consomem mais RAM. Como melhoria futura, pretendemos implementar memória RAM compartilhada para evitar que cada worker precise carregar sua própria cópia da galeria, otimizando o uso de hardware."

---

## Slide 9: Conclusão & Demonstração Prática
*   **Conteúdo Visual:**
    *   Link/QR Code do repositório Git do projeto.
    *   Demonstração do sistema aberto identificando rostos de teste na Webcam.
*   **Tópicos Chave:**
    *   Sucesso nos requisitos mínimos da disciplina.
    *   Funcionamento comprovado de ponta a ponta (Cloudflare R2 -> OpenCV -> Multiprocessing).
*   **Roteiro de Fala:**
    > "Concluindo, o Parallel Face Search cumpre com folga todos os requisitos mínimos da disciplina de Sistemas Distribuidos e Paralelos. Temos um problema claro, resolvemos usando paralelismo comprovado por medições e métricas de desempenho. Vamos agora passar para a demonstração prática da nossa webcam reconhecendo o rosto com e sem máscara em tempo real."
