const pool = require('../config/db');

async function createImageRecord({ name, category, url, timestamp }) {
    const query = `
    INSERT INTO images (name, category, url, timestamp)
    VALUES ($1, $2, $3, $4) RETURNING *;
  `;
    const values = [name, category, url, timestamp];
    const { rows } = await pool.query(query, values);
    return rows[0];
}

async function getAllImages() {
    const { rows } = await pool.query('SELECT * FROM images ORDER BY timestamp DESC');
    return rows;
}

async function getImageByName(name) {
    const { rows } = await pool.query('SELECT * FROM images WHERE name = $1', [name]);
    return rows[0];
}

async function updateImageByName(name, updateData) {
    const fields = [];
    const values = [];
    let idx = 1;

    for (const key in updateData) {
        fields.push(`${key} = $${idx++}`);
        values.push(updateData[key]);
    }
    values.push(name);

    const query = `
    UPDATE images SET ${fields.join(', ')}
    WHERE name = $${idx}
    RETURNING *;
  `;

    const { rows } = await pool.query(query, values);
    return rows[0];
}

async function deleteImageByName(name) {
    const { rows } = await pool.query('DELETE FROM images WHERE name = $1 RETURNING *', [name]);
    return rows[0];
}

module.exports = {
    createImageRecord,
    getAllImages,
    getImageByName,
    updateImageByName,
    deleteImageByName,
};
