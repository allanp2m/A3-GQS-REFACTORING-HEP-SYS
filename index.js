const express = require('express');
const path = require('path');
const { PredictionClient } = require('./src/services/PredictionClient');
const { DiagnosisRepository } = require('./src/repositories/DiagnosisRepository');

// Servidor principal Express encapsulado em classe para facilitar manutenção e testes

class AppServer {
  constructor(config) {
    this.port = config.port || 3000;
    this.predictionBaseUrl = config.predictionBaseUrl || 'http://localhost:5000';
    this.app = express();
    this.repo = new DiagnosisRepository(path.join(__dirname, 'public/diagnosticos.json'));
    this.predictionClient = new PredictionClient(this.predictionBaseUrl);
    this._configure();
    this._routes();
  }

  _configure() {
    // Middlewares essenciais
    this.app.use(express.json());
    this.app.use(express.static('public'));
  }

  _routes() {
    // Rota de predição utilizando serviço Python (Flask)
    this.app.post('/diagnose', async (req, res) => {
      try {
        const payload = req.body || {};
        const result = await this.predictionClient.diagnose(payload);
        res.json(result);
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    // Rota opcional para re-treinar o modelo
    this.app.post('/retrain', async (req, res) => {
      try {
        const info = await this.predictionClient.retrain();
        res.json(info);
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    // Listagem de diagnósticos salvos
    this.app.get('/diagnosticos', (req, res) => {
      res.json(this.repo.list());
    });

    // Criação de novo diagnóstico (salva o retorno da predição + dados)
    this.app.post('/diagnosticos', (req, res) => {
      try {
        const novo = this.repo.create(req.body || {});
        res.json(novo);
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    // Atualização de registro
    this.app.put('/diagnosticos/:id', (req, res) => {
      try {
        this.repo.update(req.params.id, req.body || {});
        res.json({ ok: true });
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    // Remoção de registro
    this.app.delete('/diagnosticos/:id', (req, res) => {
      try {
        this.repo.delete(req.params.id);
        res.json({ ok: true });
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });
  }

  start() {
    // Inicializa o servidor HTTP
    this.app.listen(this.port, () => {
      console.log(`Node.js server running on http://localhost:${this.port}`);
    });
  }
}

// Inicialização da aplicação
const server = new AppServer({
  port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
  predictionBaseUrl: process.env.PREDICTION_API || 'http://localhost:5000'
});
server.start();
