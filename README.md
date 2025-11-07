<div align="center">

# A3-GQS-REFACTORING-HEP-SYS

Sistema de diagnóstico hepático com refatoração orientada a objetos, separação de responsabilidades e serviço de predição em Python (Flask + scikit-learn) consumido por aplicação Node.js (Express).

</div>

## 🧱 Arquitetura Refatorada

| Camada | Tecnologia | Responsabilidade |
|--------|------------|------------------|
| Predição (Python) | Flask + scikit-learn | Treino e inferência do modelo KNN com pipeline de pré-processamento |
| Serviço Web Python | `model_api.py` | Endpoints `/train` e `/predict`, validação mínima, logging condicional (DEBUG) |
| Lógica de Modelo | `prediction_service.py` | Classe `HepatitisPredictor` (treino, salvar/carregar, predição) + `PredictionRepository` (log opcional em MySQL) |
| API Principal | Node.js (Express) | CRUD de diagnósticos e proxy para predição via `/diagnose` |
| Cliente de Predição | `PredictionClient.js` | Encapsula chamadas HTTP para Flask (predict/train) |
| Persistência Local | `DiagnosisRepository.js` | Armazena diagnósticos em arquivo JSON |

## 🚀 Principais Melhorias

1. Separação clara entre API Web e lógica de ML (POO no Python e JS).
2. Uso de `Pipeline` + `ColumnTransformer` para evitar divergência treino/inferência.
3. Modelo carregado em memória (evita re-load por requisição).
4. Normalização dos dados de entrada com imputação e codificação robusta.
5. Endpoints adicionais: `/train` para re-treinar e `/retrain` no Node.
6. Logs de depuração opcionais com `DEBUG=1`.
7. Preparado para MySQL via SQLAlchemy (caso `DB_URL` seja fornecida).
8. Comentários padronizados em PT-BR e código enxuto.

## 📂 Estrutura de Pastas

```
model/
	HepatitisCdata.csv      # Dataset
	model_api.py            # Flask app (endpoints)
	prediction_service.py   # Classes de predição e repositório
src/
	services/
		PredictionClient.js   # Cliente HTTP para o Flask
	repositories/
		DiagnosisRepository.js# Persistência local
index.js                  # Servidor Express (rotas principais)
public/                   # Front-end estático + diagnósticos.json
requirements.txt          # Dependências Python
package.json              # Dependências Node
```

## 🔧 Requisitos

Python 3.11+ (ideal) / 3.13 testado
Node.js 18+

## 📦 Instalação

### Backend Python (Flask)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Backend Node.js (Express)

```powershell
npm install
```

## ▶️ Execução

Em dois terminais separados:

```powershell
# Terminal 1 - Flask
$env:DEBUG="1"; python model\model_api.py

# Terminal 2 - Node
npm start
```

Aplicação web: http://localhost:3000  
Serviço de predição: http://localhost:5000

## 🔄 Endpoints Principais

### Flask
| Método | Rota | Descrição |
|--------|------|-----------|
| POST | /train | Re-treina modelo e retorna acurácia de validação |
| POST | /predict | Prediz categoria hepática para um registro |

Exemplo de payload:
```json
{
	"Age": 45,
	"Sex": "m",
	"ALB": 40.2,
	"ALP": 60.1,
	"ALT": 15.7,
	"AST": 22.3,
	"BIL": 5.1,
	"CHE": 7.2,
	"CHOL": 3.9,
	"CREA": 90,
	"GGT": 25.4,
	"PROT": 70
}
```

Resposta:
```json
{
	"prediction": 0,
	"label": "0=Blood Donor",
	"accuracy": 0.9876
}
```

### Node.js
| Método | Rota | Descrição |
|--------|------|-----------|
| POST | /diagnose | Chama Flask e retorna predição |
| POST | /retrain | Proxy para /train do Flask |
| GET | /diagnosticos | Lista diagnósticos salvos |
| POST | /diagnosticos | Cria registro (usuário + resultado) |
| PUT | /diagnosticos/:id | Atualiza registro |
| DELETE | /diagnosticos/:id | Remove registro |

## 🗄️ Persistência Opcional (MySQL)

Defina a variável de ambiente:
```
DB_URL=mysql+pymysql://usuario:senha@host:3306/banco
```
Predições serão logadas na tabela `predictions` (criada automaticamente).

## 🧪 Teste Rápido via Python

```python
from prediction_service import HepatitisPredictor, make_default_paths
p = HepatitisPredictor(make_default_paths())
p.train()
print(p.predict({"Age":32,"Sex":"m","ALB":38.5,"ALP":52.5,"ALT":7.7,"AST":22.1,"BIL":7.5,"CHE":6.93,"CHOL":3.23,"CREA":106,"GGT":12.1,"PROT":69}))
```

## 🔍 Debug

Exportar `DEBUG=1` habilita logs adicionais e traceback nos erros.

## ✅ Próximos Passos (Sugestões)

- Implementar testes automatizados (Pytest / Jest)
- Adicionar validação formal (Pydantic / Zod)
- Containerização (Dockerfile + docker-compose)
- Autenticação/JWT para rotas de diagnóstico
- Versionar modelos e métricas de treino
- Substituir JSON plano por SQLite/MySQL para registros de diagnósticos

## 📄 Licença

Uso acadêmico / estudo. Adaptar conforme necessidade institucional.

---
Refatoração concluída: arquitetura modular, comentários claros e expansão facilitada. 
Se precisar de testes ou Docker, abra uma issue ou solicite diretamente.
