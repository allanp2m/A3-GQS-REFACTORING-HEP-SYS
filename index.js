const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const app = express();

const DB_FILE = path.join(__dirname, 'public/diagnosticos.json');

app.use(express.json());
app.use(express.static('public'));

// Diagnóstico com IA
app.post('/diagnose', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/predict', req.body);
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Carregar diagnósticos do arquivo
const carregarDiagnosticos = () => {
  try {
    if (!fs.existsSync(DB_FILE)) return [];
    return JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
  } catch (err) {
    console.error("Erro ao carregar diagnosticos.json:", err.message);
    return [];
  }
};


// Salvar diagnósticos
const salvarDiagnosticos = (data) => {
  fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));
};

// Criar novo diagnóstico
app.post('/diagnosticos', (req, res) => {
  const lista = carregarDiagnosticos();
  const novo = { id: Date.now(), ...req.body };
  lista.push(novo);
  salvarDiagnosticos(lista);
  res.json(novo);
});

// Listar todos
app.get('/diagnosticos', (req, res) => {
  res.json(carregarDiagnosticos());
});

// Editar
app.put('/diagnosticos/:id', (req, res) => {
  let lista = carregarDiagnosticos();
  const id = parseInt(req.params.id);
  lista = lista.map(d => d.id === id ? { ...d, ...req.body } : d);
  salvarDiagnosticos(lista);
  res.json({ ok: true });
});

// Deletar
app.delete('/diagnosticos/:id', (req, res) => {
  let lista = carregarDiagnosticos();
  const id = parseInt(req.params.id);
  lista = lista.filter(d => d.id !== id);
  salvarDiagnosticos(lista);
  res.json({ ok: true });
});

app.listen(3000, () => {
  console.log('Node.js server running on http://localhost:3000');
});
