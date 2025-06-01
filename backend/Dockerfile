# Backlog Builder Backend Service
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Create necessary directories with correct permissions
RUN mkdir -p /app/logs && \
    chown -R node:node /app

# Copy package files
COPY --chown=node:node package*.json ./


# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY --chown=node:node src/ ./src/
COPY --chown=node:node *.js ./

# Set environment variables
ENV NODE_ENV=production

# Switch to non-root user
USER node

# Expose the app port
EXPOSE 3000

# Start the application
CMD ["node", "src/app.js"]
