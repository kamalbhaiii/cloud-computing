const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const swaggerUi = require('swagger-ui-express');
const fs = require('fs');
const path = require('path');

const env = require('./config/env.json');
const logger = require('./middlewares/logger');
const errorHandler = require('./middlewares/errorHandler');
const imageRoutes = require('./routes/imageRoutes');
const minIORoutes = require('./routes/minIORoutes');
const minioService = require('./services/minioService');
const db = require('./config/db');

const app = express();

app.use(cors());
app.use(helmet());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(logger);

app.use('/api/images', imageRoutes);
app.use('/api/minio', minIORoutes);
app.get('/api/test', (req,res) => {
    res.status(200).send({
        message:"Backend is live!"
    })
})

const swaggerDocument = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'swagger', 'swagger.json'))
);

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

app.use(errorHandler);

const PORT = env.PORT || 5000;

async function startServer() {
    const minioOk = await minioService.checkConnection();
    const dbOk = await db.checkConnection();

    console.log(`MinIO connection: ${minioOk ? '✅ Connected' : '❌ Failed'}`);
    console.log(`PostgreSQL connection: ${dbOk ? '✅ Connected' : '❌ Failed'}`);

    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
    });
}

startServer();
