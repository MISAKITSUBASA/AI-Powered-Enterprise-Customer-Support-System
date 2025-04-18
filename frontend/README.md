# AI Customer Support System - Frontend

This is the React frontend for the AI-Powered Enterprise Customer Support System. It provides an intuitive interface for users to interact with the AI assistant and for administrators to manage the system.

## Features

- **Responsive Chat Interface**: Modern, intuitive chat experience
- **Emotion Analysis Visualization**: Visual indicators for detected user emotions
- **Admin Dashboard**: Comprehensive analytics and system management
- **Knowledge Base Management**: Upload and manage documents
- **User Authentication**: Secure login and registration

## Project Structure

```
frontend/
├── src/                    # Source code
│   ├── components/         # Reusable UI components
│   ├── context/            # React context for state management
│   ├── pages/              # Main page components
│   │   ├── Admin.js        # Admin dashboard
│   │   ├── Chat.js         # Chat interface
│   │   ├── Login.js        # Login page
│   │   └── Register.js     # Registration page
│   └── styles/             # CSS stylesheets
├── public/                 # Static assets
└── build/                  # Production build output
```

## Environment Setup

The frontend expects the backend API to be available. In development mode, API requests are proxied to the backend server.

## Docker Deployment

The frontend is containerized and can be deployed using Docker:

```bash
# Build the Docker image
docker build -t customer-support-frontend .

# Run the container
docker run -p 3000:80 customer-support-frontend
```

The Dockerfile uses nginx to serve the static assets in production.

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
