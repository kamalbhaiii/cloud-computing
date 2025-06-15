const express = require('express');
const router = express.Router();
const minIOController = require('../controllers/minIOController');

router.get('/', minIOController.getMinIOImage);

module.exports = router;