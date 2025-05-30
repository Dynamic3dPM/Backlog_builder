const path = require('path');
const mime = require('mime-types');

/**
 * Validate uploaded audio file
 * @param {Object} file - Multer file object
 * @returns {Object} Validation result
 */
function validateFileUpload(file) {
    if (!file) {
        return {
            isValid: false,
            error: 'No file provided'
        };
    }

    // Check file size (max 500MB)
    const maxSize = parseInt(process.env.MAX_AUDIO_SIZE_MB || '500') * 1024 * 1024;
    if (file.size > maxSize) {
        return {
            isValid: false,
            error: `File size exceeds maximum allowed size of ${maxSize / (1024 * 1024)}MB`
        };
    }

    // Check file extension
    const allowedExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'];
    const fileExtension = path.extname(file.originalname).toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
        return {
            isValid: false,
            error: `Unsupported file format: ${fileExtension}. Allowed formats: ${allowedExtensions.join(', ')}`
        };
    }

    // Check MIME type
    const allowedMimeTypes = [
        'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac', 
        'audio/ogg', 'audio/wma', 'audio/aac', 'audio/mp3'
    ];
    
    const detectedMimeType = mime.lookup(file.originalname);
    if (file.mimetype && !allowedMimeTypes.includes(file.mimetype) && 
        !allowedMimeTypes.includes(detectedMimeType)) {
        return {
            isValid: false,
            error: `Invalid MIME type: ${file.mimetype}. Expected audio format.`
        };
    }

    // Check file name for security
    const filename = file.originalname;
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
        return {
            isValid: false,
            error: 'Invalid filename: contains path traversal characters'
        };
    }

    return {
        isValid: true,
        fileInfo: {
            name: filename,
            size: file.size,
            extension: fileExtension,
            mimeType: file.mimetype || detectedMimeType
        }
    };
}

/**
 * Express middleware for file validation
 */
function fileValidationMiddleware(req, res, next) {
    if (!req.file && !req.files) {
        return res.status(400).json({
            success: false,
            error: 'No file uploaded'
        });
    }

    const files = req.files || [req.file];
    
    for (const file of files) {
        const validation = validateFileUpload(file);
        if (!validation.isValid) {
            return res.status(400).json({
                success: false,
                error: validation.error,
                filename: file.originalname
            });
        }
    }

    next();
}

module.exports = {
    validateFileUpload,
    fileValidationMiddleware
};