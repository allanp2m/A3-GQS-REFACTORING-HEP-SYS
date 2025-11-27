const axios = require('axios');
const { PredictionClient } = require('./PredictionClient');

jest.mock('axios');

describe('PredictionClient', () => {
  const baseUrl = 'http://localhost:5000';

  test('diagnose envia POST para /predict e retorna dados', async () => {
    const client = new PredictionClient(baseUrl);
    const payload = { x: 10 };

    axios.post.mockResolvedValue({ data: { ok: true } });

    const result = await client.diagnose(payload);

    expect(axios.post).toHaveBeenCalledWith(
      `${baseUrl}/predict`,
      payload,
      { timeout: 8000 }
    );

    expect(result).toEqual({ ok: true });
  });

  test('diagnose lança erro se payload for inválido', async () => {
    const client = new PredictionClient(baseUrl);

    await expect(client.diagnose(null)).rejects.toThrow();
    await expect(client.diagnose("aaa")).rejects.toThrow();
  });

  test('retrain envia POST para /train e retorna dados', async () => {
    const client = new PredictionClient(baseUrl);

    axios.post.mockResolvedValue({ data: { retrained: true } });

    const result = await client.retrain();

    expect(axios.post).toHaveBeenCalledWith(
      `${baseUrl}/train`,
      {},
      { timeout: 20000 }
    );

    expect(result).toEqual({ retrained: true });
  });

  test('retrain lança erro se axios falhar', async () => {
    const client = new PredictionClient(baseUrl);

    axios.post.mockRejectedValue(new Error('erro'));

    await expect(client.retrain()).rejects.toThrow(
      'Falha ao re-treinar modelo: erro'
    );
  });
});
