// backend/routes/whatsapp.js
const express = require('express');
const router = express.Router();

// CLOUD CRITICAL: Use Environment Variable instead of hardcoded 127.0.0.1
const RAG_URL = process.env.RAG_ENGINE_URL || 'http://127.0.0.1:5001';

router.post('/webhook', async (req, res) => {
  const incomingMsg = req.body.Body;
  
  if (!incomingMsg) {
    return res.status(200).send('<Response></Response>');
  }

  console.log(`\n📱 WhatsApp Message Received: ${incomingMsg}`);

  try {
    const aiResponse = await fetch(`${RAG_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: incomingMsg })
    });
    
    const aiData = await aiResponse.json();
    const finalAnswer = aiData.answer || "I'm sorry, I couldn't process that.";

    const twiml = `
      <Response>
        <Message>${finalAnswer}</Message>
      </Response>
    `;

    res.set('Content-Type', 'text/xml');
    res.status(200).send(twiml);

  } catch (error) {
    console.error('❌ Error connecting to Python AI Engine:', error.message);
    const errorTwiml = `
      <Response>
        <Message>The UPGCS Assistant is currently offline for maintenance. 🎓</Message>
      </Response>
    `;
    res.set('Content-Type', 'text/xml');
    res.status(200).send(errorTwiml);
  }
});

router.get('/status', (req, res) => {
  res.json({
    status: 'WhatsApp route active 🚀',
    note: 'Connected to Python RAG Engine'
  });
});

module.exports = router;