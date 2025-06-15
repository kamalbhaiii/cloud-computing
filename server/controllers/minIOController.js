const axios = require('axios')

async function getMinIOImage (req, res) {
    const imageUrl = req.query.url;

    if (!imageUrl) {
        return res.status(400).send("Missing 'url' query parameter.");
    }

    try {
        const response = await axios.get(imageUrl, {
            responseType: 'stream'
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);
    } catch (error) {
        console.error("Failed to fetch image:", error.message);
        res.status(500).send("Error fetching image from MinIO.");
    }
}

module.exports = {
    getMinIOImage
}