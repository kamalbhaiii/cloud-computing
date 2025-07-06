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
        const timestamp = dayjs().toISOString();
        const uniqueName = `${category}_${timestamp}}`;
        const url = await minioService.uploadImage(file.buffer, uniqueName, file.mimetype);

        const record = await imageModel.createImageRecord({
            name: uniqueName,
            category: category,
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

        const proxiedImages = images.map((img) => {
            return {
                ...img,
                url: `${req.protocol}://server.local/api/minio?url=${encodeURIComponent(img.url)}`
            };
        });

        res.status(200).json(success(proxiedImages));
    } catch (err) {
        res.status(500).json(error('Failed to fetch images'));
    }
}

async function updateImage(req, res) {
    const { name } = req.params;
    const { category } = req.body;

    try {
        const existing = await imageModel.getImageByName(name);
        if (!existing) {
            return res.status(404).json(error('Image not found', 404));
        }

        const updated = await imageModel.updateImageByName(name, { category });
        const imageBuffer = await minioService.getImageBuffer(name); // You need to implement this
        sendImageNotificationToTelegram(imageBuffer, `Updated category: ${category}`);

        res.status(200).json(success(updated, 'Image metadata updated'));
    } catch (err) {
        console.error(err);
        res.status(500).json(error('Failed to update image metadata'));
    }
}

async function deleteImage(req, res) {
    const { names } = req.body;

    if (!Array.isArray(names) || names.length === 0) {
        return res.status(400).json(error('Image name(s) must be a non-empty array', 400));
    }

    const results = {
        deleted: [],
        notFound: [],
        failed: [],
    };

    for (const name of names) {
        try {
            const existing = await imageModel.getImageByName(name);
            if (!existing) {
                results.notFound.push(name);
                continue;
            }

            await minioService.deleteImage(name);
            await imageModel.deleteImageByName(name);
            results.deleted.push(name);
        } catch (err) {
            results.failed.push({ name, error: err.message || 'Unknown error' });
        }
    }

    const status = results.failed.length > 0 ? 207 : 200;
    return res.status(status).json(success(results, 'Image deletion processed'));
}

module.exports = {
    uploadImage,
    getImages,
    updateImage,
    deleteImage
};
