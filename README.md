# A3-GQS-REFACTORING-HEP-SYS

# 📌 Análise de Arquitetura e Boas Práticas

## 🔎 Visão Geral
O projeto apresenta **forte acoplamento** entre o pipeline de treino e a API de predição, ausência de camadas bem definidas e práticas inadequadas de configuração, validação e operação.  
Principais problemas identificados:

- **Acoplamento excessivo**: o treinamento do modelo acontece dentro do mesmo módulo que expõe a rota, dificultando manutenção, implantação e escalabilidade.
- **Ausência de POO e camadas**: lógica de negócio, I/O e web estão misturados sem separação em serviços, repositórios ou controladores.
- **Configuração hardcoded**: caminhos, portas e URLs fixos no código em vez de variáveis de ambiente.
- **Validação de entrada inexistente**: não há schemas formais, aumentando risco de erros e inconsistência de dados.
- **Sem testes, logs estruturados ou métricas**: dificulta depuração e evolução.
- **Problemas operacionais**: dois serviços (Flask e Express) sem orquestração, healthcheck ou retry/backoff.

## ⚙️ model_api.py (Flask + sklearn)

### Problemas Identificados
- **Treinamento no import**: modelo é treinado e serializado na importação → inicialização lenta, mistura de responsabilidades e dificuldade para escalar.
- **Data leakage**: imputação de NaN e label encoding feitos antes do `train_test_split`.
- **Pré-processamento inconsistente**: inferência não replica o fluxo de treino (ex.: ausência de imputação).
- **Uso incorreto do LabelEncoder**: introduz ordinalidade artificial → deveria ser One-Hot.
- **Dependência manual da ordem das colunas** → solução frágil.
- **Não uso de `Pipeline` do sklearn** → serialização manual de modelo/scaler/encoders.
- **Carregamento do modelo em cada requisição (`joblib.load`)** → overhead de I/O e latência.
- **Métrica mal nomeada**: campo `accuracy` retorna probabilidade prevista, não acurácia real.
- **Tratamento de erros genérico** → risco de vazar informações internas.
- **Leitura de CSV frágil**: caminho relativo fixo; `drop` em coluna sem checar existência.
- **Servidor inadequado para produção**: uso de `app.run` em vez de WSGI (ex.: gunicorn).
- **Ausência de avaliação de desempenho**: `X_test/y_test` não são usados; sem métricas offline.
- **Reprodutibilidade fraca**: apenas `random_state`; versões de libs não pinadas; sem seed global.

## ⚙️ index.js (Express + armazenamento em arquivo)

### Problemas Identificados
- **Armazenamento inseguro**: `diagnosticos.json` em `/public` → exposição pública de dados sensíveis.
- **Operações síncronas de I/O**: `readFileSync` e `writeFileSync` → bloqueiam o event loop.
- **Condições de corrida**: padrão read-modify-write sem bloqueio → risco de sobrescrita.
- **Validação inexistente**: payloads aceitos sem checagem; endpoints retornam sempre sucesso.
- **Geração de IDs frágil**: uso de `Date.now()` → não garante unicidade sob concorrência.
- **Acoplamento ao serviço Python**: URL hardcoded, sem timeout/retry.
- **Falta de segurança**: sem autenticação, rate limiting ou sanitização de dados.
- **Organização ruim**: lógica de negócio embutida nos handlers; ausência de classes/serviços reutilizáveis.
- **Status codes inconsistentes**: respostas de erro retornam `200 OK`.
- **Logs precários**: apenas `console.log`, sem middleware ou correlação.

## 🚨 Impactos Práticos

- **Manutenibilidade baixa**: mudanças quebram facilmente o pipeline.
- **Riscos de segurança e privacidade**: exposição de diagnósticos publicamente.
- **Baixo desempenho sob carga**: I/O síncrono no Node e recarregamento do modelo em cada request no Flask.
- **Confiabilidade fraca**: falta de validações, erros genéricos, ausência de testes.
- **Comprometimento científico**: métricas inválidas devido a data leakage e definição incorreta de “accuracy”.
