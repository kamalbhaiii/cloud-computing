const express = require('express');
const router = express.Router();
const multer = require('multer');
const { body, param } = require('express-validator');

const upload = multer();
const imageController = require('../controllers/imageController');

router.post(
    '/',
    upload.single('image'),
    body('category').isIn(['Dog', 'Cat', 'Bird', 'Person']),
    imageController.uploadImage
);

router.get('/', imageController.getImages);

router.put(
    '/:name',
    param('name').notEmpty(),
    body('category').optional().isIn(['Dog', 'Cat', 'Bird', 'Person']),
    imageController.updateImage
);

router.delete(
    '/:name',
    param('name').notEmpty(),
    imageController.deleteImage
);

module.exports = router;
