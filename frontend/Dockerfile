# Build stage
FROM node:19-alpine AS build

WORKDIR /app

# Copy package.json and install dependencies first for better caching
COPY package*.json ./
RUN npm install

# Copy app source and build
COPY . ./
RUN npm run build

# Production stage
FROM nginx:1.23-alpine

# Copy built files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Add nginx configuration for React Router
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]