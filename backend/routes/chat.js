// backend/routes/chat.js
const express = require('express');
const router = express.Router();
const axios = require('axios');
const { formatResponse } = require('../utils/response');

const RAG_URL = process.env.RAG_ENGINE_URL || 'http://localhost:5001';

router.post('/', async (req, res) => {
  try {
    const { message } = req.body;

    // ── 1. Strict Input Validation ───────────────────────────
    if (!message || typeof message !== 'string') {
      return res.status(400).json(
        formatResponse(false, null, 'Please provide a valid text message.')
      );
    }

    const cleanMessage = message.trim();

    // Prevent empty or absurdly long inputs (saves RAM/API tokens)
    if (cleanMessage.length < 2) {
      return res.status(400).json(
        formatResponse(false, null, 'Message is too short.')
      );
    }
    if (cleanMessage.length > 500) {
      return res.status(400).json(
        formatResponse(false, null, 'Message exceeds 500 characters. Please be brief.')
      );
    }

    console.log(`📩 Validated User Query: "${cleanMessage}"`);

    // ── 2. Call Python RAG Engine ────────────────────────────
    const ragResponse = await axios.post(
      `${RAG_URL}/query`,
      { question: cleanMessage },
      { timeout: 90000 }
    );

    const { answer, sources, confidence } = ragResponse.data;

    // ── 3. Return Standardized Success ───────────────────────
    return res.status(200).json(
      formatResponse(true, {
        question: cleanMessage,
        answer: answer || 'I could not find an answer.',
        sources: sources || [],
        confidence: confidence || 0
      })
    );

  } catch (error) {
    console.error('❌ Routing Error:', error.message);

    // Handle Python server offline gracefully
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json(
        formatResponse(false, null, 'AI Engine is currently offline. Please start the Python server.')
      );
    }

    // Handle timeouts
    if (error.code === 'ECONNABORTED') {
       return res.status(504).json(
         formatResponse(false, null, 'AI Engine took too long to respond.')
       );
    }

    return res.status(500).json(
      formatResponse(false, null, 'Internal server error while processing your request.')
    );
  }
});

router.get('/health', async (req, res) => {
  try {
    await axios.get(`${RAG_URL}/health`, { timeout: 5000 });
    res.json(formatResponse(true, { backend: 'Secured & Running', ragEngine: 'Online' }));
  } catch {
    res.json(formatResponse(true, { backend: 'Secured & Running', ragEngine: 'Offline' }));
  }
});

module.exports = router;