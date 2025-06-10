function success(data, message = 'Success') {
    return {
        status: 'success',
        message,
        data
    };
}

function error(message = 'An error occurred', code = 500) {
    return {
        status: 'error',
        message,
        code
    };
}

module.exports = { success, error };
  