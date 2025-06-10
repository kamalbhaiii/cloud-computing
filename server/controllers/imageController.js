const { validationResult } = require('express-validator');
const dayjs = require('dayjs');
const { v4: uuidv4 } = require('uuid');
const imageModel = require('../models/imageModel');
const minioService = require('../services/minioService');
const { success, error } = require('../utils/response');
const { sendImageNotificationToTelegram } = require('../services/telegramService');

async function uploadImage(req, res) {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json(error('Invalid request parameters', 400));
    }

    const { category } = req.body;
    const file = req.file;

    if (!file) {
        return res.status(400).json(error('Image file is required', 400));
    }

    try {
        const uniqueName = `${uuidv4()}_${file.originalname}`;
        const url = await minioService.uploadImage(file.buffer, uniqueName, file.mimetype);
        const timestamp = dayjs().toISOString();

        const record = await imageModel.createImageRecord({
            name: [uniqueName],
            category: [category],
            url,
            timestamp
        });

        res.status(201).json(success(record, 'Image uploaded and metadata saved'));

        sendImageNotificationToTelegram(file.buffer, `Detected: ${category}`)
    } catch (err) {
        console.log(err)
        res.status(500).json(error('Failed to upload image'));
    }
}

async function getImages(req, res) {
    try {
        const images = await imageModel.getAllImages();
        res.status(200).json(success(images));
    } catch (err) {
        res.status(500).json(error('Failed to fetch images'));
    }
}

async function updateImage(req, res) {
    const { name } = req.params;
    const { category } = req.body;

    try {
        const existing = await imageModel.getImageByName(name);
        if (!existing) return res.status(404).json(error('Image not found', 404));

        const updated = await imageModel.updateImageByName(name, { category });
        res.status(200).json(success(updated, 'Image metadata updated'));
    } catch (err) {
        res.status(500).json(error('Failed to update image metadata'));
    }
}

async function deleteImage(req, res) {
    const { name } = req.params;

    try {
        const existing = await imageModel.getImageByName([name]);
        if (!existing) return res.status(404).json(error('Image not found', 404));

        await minioService.deleteImage(name);
        await imageModel.deleteImageByName([name]);

        res.status(200).json(success(null, 'Image deleted successfully'));
    } catch (err) {
        res.status(500).json(error('Failed to delete image'));
    }
}

module.exports = {
    uploadImage,
    getImages,
    updateImage,
    deleteImage
};
