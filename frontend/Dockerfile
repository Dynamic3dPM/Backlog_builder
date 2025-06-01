FROM node:16

# Create app directory
WORKDIR /usr/src/app

# Install app dependencies
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app source
COPY . .

# Expose port for Vue dev server
EXPOSE 8080

# Start the app with cache disabled
ENV BABEL_DISABLE_CACHE=1
ENV ESLINT_DISABLE_CACHE=1 
ENV NODE_OPTIONS=--max_old_space_size=4096

# Start with no permissions issues
CMD ["sh", "-c", "chmod -R 777 /usr/src/app/node_modules/.cache || mkdir -p /usr/src/app/node_modules/.cache && chmod -R 777 /usr/src/app/node_modules/.cache && npm run serve"]
