const axios = require('axios');
const FormData = require('form-data');
const env = require('../config/env.json');

const TELEGRAM_BOT_TOKEN = env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_IDS = env.TELEGRAM_CHAT_IDS;

async function sendImageNotificationToTelegram(buffer, caption) {
    for (const TELEGRAM_CHAT_ID of TELEGRAM_CHAT_IDS) {
        const formData = new FormData();
        formData.append('chat_id', TELEGRAM_CHAT_ID.toString());
        formData.append('caption', caption);
        formData.append('photo', buffer, {
            filename: 'image.jpg',
            contentType: 'image/jpeg',
        });

        try {
            await axios.post(
                `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendPhoto`,
                formData,
                { headers: formData.getHeaders() }
            );
            console.log(`✅ Notification sent to user: ${TELEGRAM_CHAT_ID}`);
        } catch (err) {
            console.error(`❌ Failed to send to ${TELEGRAM_CHAT_ID}:`, err.message);
        }
    }
}

module.exports = {
    sendImageNotificationToTelegram
};
