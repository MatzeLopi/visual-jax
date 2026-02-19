FROM node:25-bookworm

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . . 


EXPOSE 3000

CMD ["npm", "run", "dev"]