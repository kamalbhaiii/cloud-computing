const { Client } = require('minio');
const env = require('../config/env.json');

const minioClient = new Client({
    endPoint: env.MINIO_END_POINT,
    port: env.MINIO_PORT,
    accessKey: env.MINIO_ACCESS_KEY,
    secretKey: env.MINIO_SECRET_KEY,
    useSSL: env.MINIO_USE_SSL
});

const policy = {
    Version: "2012-10-17",
    Statement: [{
        Effect: "Allow",
        Principal: { AWS: ["*"] },
        Action: ["s3:GetObject"],
        Resource: [`arn:aws:s3:::${env.MINIO_BUCKET}/*`]
    }]
};

minioClient.setBucketPolicy(env.MINIO_BUCKET, JSON.stringify(policy), function (err) {
    if (err) return console.log("Error setting policy:", err);
    console.log("Public read policy applied.");
});

async function uploadImage(buffer, name, mimetype) {
    await minioClient.putObject(env.MINIO_BUCKET, name, buffer, { 'Content-Type': mimetype });
    return `${env.MINIO_SSL ? 'https' : 'http'}://${env.MINIO_END_POINT}:${env.MINIO_PORT}/${env.MINIO_BUCKET}/${name}`;
}

async function deleteImage(name) {
    await minioClient.removeObject(env.MINIO_BUCKET, name);
}

async function checkConnection() {
    try {
        const buckets = await minioClient.listBuckets();
        return buckets.map(b => b.name).includes(env.MINIO_BUCKET);
    } catch (err) {
        return false;
    }
}

module.exports = {
    uploadImage,
    deleteImage,
    checkConnection
};
