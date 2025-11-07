const fs = require('fs');
const path = require('path');

// Repositório simples baseado em arquivo JSON local
class DiagnosisRepository {
  constructor(filePath) {
    this.filePath = filePath;
    this._ensureStorage(); // Garante existência do arquivo e diretório
  }

  // Cria diretório e arquivo caso ausentes
  _ensureStorage() {
    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    if (!fs.existsSync(this.filePath)) {
      fs.writeFileSync(this.filePath, '[]', 'utf8');
    }
  }

  // Lê todos os registros
  _readAll() {
    try {
      const txt = fs.readFileSync(this.filePath, 'utf8');
      const data = JSON.parse(txt);
      return Array.isArray(data) ? data : [];
    } catch (err) {
      console.error('Erro ao ler diagnosticos.json:', err.message);
      return [];
    }
  }

  // Escreve lista completa sobrescrevendo arquivo
  _writeAll(list) {
    fs.writeFileSync(this.filePath, JSON.stringify(list, null, 2));
  }

  // Lista registros
  list() {
    return this._readAll();
  }

  // Cria novo registro
  create(doc) {
    const list = this._readAll();
    const novo = { id: Date.now(), ...doc };
    list.push(novo);
    this._writeAll(list);
    return novo;
  }

  // Atualiza registro
  update(id, patch) {
    const list = this._readAll();
    const key = parseInt(id);
    const updated = list.map((d) => (d.id === key ? { ...d, ...patch } : d));
    this._writeAll(updated);
    return true;
  }

  // Remove registro
  delete(id) {
    const list = this._readAll();
    const key = parseInt(id);
    const filtered = list.filter((d) => d.id !== key);
    this._writeAll(filtered);
    return true;
  }
}

module.exports = { DiagnosisRepository };