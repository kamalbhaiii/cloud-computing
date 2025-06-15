const express = require('express');
const router = express.Router();
const minIOController = require('../controllers/minIOController');
const {query} = require('express-validator')

router.get('/', query("url"), minIOController.getMinIOImage);

module.exports = router;