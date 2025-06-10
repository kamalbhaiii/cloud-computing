const { Pool } = require('pg');
const env = require('./env.json');

const pool = new Pool({
    host: env.DB_HOST,
    user: env.DB_USER,
    password: env.DB_PASSWORD,
    database: env.DB_NAME,
    port: env.DB_PORT
});

async function checkConnection() {
    try {
        await pool.query('SELECT NOW()');
        return true;
    } catch {
        return false;
    }
}

module.exports = pool;
module.exports.checkConnection = checkConnection;
