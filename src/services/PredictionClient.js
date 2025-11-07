const axios = require('axios');

// Cliente HTTP para o serviço Flask de predição
class PredictionClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl; // Normaliza URL base
  }

  // Envia payload para gerar diagnóstico
  async diagnose(payload) {
    if (!payload || typeof payload !== 'object') {
      throw new Error('Payload inválido para predição.');
    }
    try {
      const res = await axios.post(`${this.baseUrl}/predict`, payload, { timeout: 8000 });
      return res.data;
    } catch (err) {
      throw new Error(`Falha ao chamar serviço de predição: ${err.message}`);
    }
  }

  // Solicita re-treino do modelo no backend Python
  async retrain() {
    try {
      const res = await axios.post(`${this.baseUrl}/train`, {}, { timeout: 20000 });
      return res.data;
    } catch (err) {
      throw new Error(`Falha ao re-treinar modelo: ${err.message}`);
    }
  }
}

module.exports = { PredictionClient };