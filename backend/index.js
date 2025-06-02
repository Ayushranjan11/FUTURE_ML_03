const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ error: 'No message provided' });
  }

  const scriptPath = path.join(__dirname, '..');

  const options = {
    mode: 'text',
    pythonPath: 'python3', // Explicitly tells the server to use python3
    pythonOptions: ['-u'],
    scriptPath: scriptPath,
    args: [message],
  };

  PythonShell.run('predict.py', options).then((results) => {
    try {
      const prediction = JSON.parse(results[0]);
      res.json(prediction);
    } catch (parseErr) {
      res.status(500).json({ error: 'Failed to parse prediction.' });
    }
  }).catch((pyErr) => {
      console.error('Error running python script:', pyErr);
      res.status(500).json({ error: 'Python script crashed.', details: pyErr.message });
  });
});

const PORT = 5001;
app.listen(PORT, () => {
  console.log(`Backend server is running on http://localhost:${PORT}`);
});