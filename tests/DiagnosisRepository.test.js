const fs = require('fs');
const path = require('path');
const { DiagnosisRepository } = require('../DiagnosisRepository');

jest.mock('fs');

describe('DiagnosisRepository', () => {
  const fakePath = path.join(__dirname, 'diagnosticos-test.json');

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('_ensureStorage cria diretório e arquivo quando não existem', () => {
    fs.existsSync.mockReturnValue(false); 

    const mkdirSpy = fs.mkdirSync.mockImplementation(() => {});
    const writeSpy = fs.writeFileSync.mockImplementation(() => {});

    new DiagnosisRepository(fakePath);

    expect(mkdirSpy).toHaveBeenCalled();
    expect(writeSpy).toHaveBeenCalledWith(fakePath, '[]', 'utf8');
  });

  test('_readAll retorna array quando JSON válido', () => {
    const data = [{ id: 1, name: 'Test' }];
    fs.readFileSync.mockReturnValue(JSON.stringify(data));

    const repo = new DiagnosisRepository(fakePath);
    const res = repo._readAll();

    expect(res).toEqual(data);
  });

  test('_readAll retorna [] quando JSON inválido ou erro', () => {
    fs.readFileSync.mockImplementation(() => {
      throw new Error('Erro de leitura');
    });

    const repo = new DiagnosisRepository(fakePath);
    const res = repo._readAll();

    expect(res).toEqual([]);
  });

  test('list() retorna registros lidos', () => {
    const mockList = [{ id: 1 }];
    fs.readFileSync.mockReturnValue(JSON.stringify(mockList));

    const repo = new DiagnosisRepository(fakePath);
    expect(repo.list()).toEqual(mockList);
  });

  test('create() adiciona novo item e salva no arquivo', () => {
    fs.readFileSync.mockReturnValue('[]'); 

    const writeSpy = fs.writeFileSync.mockImplementation(() => {});
    const repo = new DiagnosisRepository(fakePath);

    const novo = repo.create({ nome: 'Amanda' });

    expect(writeSpy).toHaveBeenCalled();
    expect(novo).toHaveProperty('id');
    expect(novo.nome).toBe('Amanda');
  });

  test('update() altera o registro correto e salva', () => {
    const initial = [{ id: 123, nome: 'A' }];
    fs.readFileSync.mockReturnValue(JSON.stringify(initial));

    const writeSpy = fs.writeFileSync.mockImplementation(() => {});
    const repo = new DiagnosisRepository(fakePath);

    repo.update(123, { nome: 'Atualizado' });

    const saved = JSON.parse(writeSpy.mock.calls[0][1]);
    expect(saved[0].nome).toBe('Atualizado');
  });

  test('update() ignora IDs que não existem', () => {
    const initial = [{ id: 1, nome: 'X' }];
    fs.readFileSync.mockReturnValue(JSON.stringify(initial));

    const writeSpy = fs.writeFileSync.mockImplementation(() => {});
    const repo = new DiagnosisRepository(fakePath);

    repo.update(999, { nome: 'Novo' });

    const saved = JSON.parse(writeSpy.mock.calls[0][1]);
    expect(saved).toEqual(initial); 
  });

  test('delete() remove item correto', () => {
    const initial = [
      { id: 1, nome: 'A' },
      { id: 2, nome: 'B' },
    ];
    fs.readFileSync.mockReturnValue(JSON.stringify(initial));

    const writeSpy = fs.writeFileSync.mockImplementation(() => {});
    const repo = new DiagnosisRepository(fakePath);

    repo.delete(2);

    const saved = JSON.parse(writeSpy.mock.calls[0][1]);
    expect(saved).toEqual([{ id: 1, nome: 'A' }]);
  });

  test('delete() não altera nada se id não existe', () => {
    const initial = [{ id: 1, nome: 'A' }];
    fs.readFileSync.mockReturnValue(JSON.stringify(initial));

    const writeSpy = fs.writeFileSync.mockImplementation(() => {});
    const repo = new DiagnosisRepository(fakePath);

    repo.delete(999);

    const saved = JSON.parse(writeSpy.mock.calls[0][1]);
    expect(saved).toEqual(initial);
  });
});
