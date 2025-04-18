# AI-Powered Enterprise Customer Support System

A scalable, AI-driven customer support platform that uses natural language processing to provide instant, accurate responses to customer inquiries while seamlessly escalating complex issues to human agents.

## System Architecture

![System Architecture](https://mermaid.ink/img/pako:eNqVksFu2zAMhl_F0GnDgCRtd5h1KDDsmLZAgR62Q0EL9CZLdChMkkzJQbPC774kbpKhQ4sJhyD8-dP8KZIN53UBbMVbwYsMhDQcbYtgj1ptNFhzTIKjAJvCgdbKvN-BNS2STsqydgEzX4mlV3DF2dbjU2dMIdXOxewgNSWVpT2qfaVN74-AQWkhUP9N3mNssdPDOe9QPZrAeCHrtKASlVu6qcGqKCH5v2aL05ZvJv3UZJuHJmsueSl29BVnYwOHGvJ0P15cLcfLnwmcwP9cLx8_Ljk5Qx3pJkZOIaLfjYu-JT8eEmT3dqzQ6ZIjmw9XC0rGD93yBcHM1lpOG_S6nsCVAJNJ3Wy9s6b8Ec8W2tXQG-uuqRWnZM5_XNIiJK1PQCZ53c3yvhtkDUgHZg8-TaOJp0-GnIPVR3rDilKDk1LWPOiAoRXqUl6lJvVE0ZfLUoTDf6-MV2g9g-HBi9blX7NmXa1Ar2ilYNbxFpA_h5Sj6Fq-srGxXY97rUX8_Q3UlMRY)

The system consists of the following components:

1. **Frontend**
   - React-based single-page application
   - User authentication and session management
   - Real-time chat interface with emotion detection visualization
   - Admin dashboard for system monitoring and knowledge base management

2. **Backend**
   - FastAPI server handling API requests
   - JWT-based authentication
   - OpenAI integration for natural language processing
   - Vector database (FAISS) for knowledge retrieval
   - Emotion analysis for adaptive responses
   - Escalation logic for complex inquiries

3. **Database**
   - PostgreSQL for structured data storage
   - SQLAlchemy ORM for database operations
   - Redis for response caching and session management

4. **Infrastructure**
   - Docker containerization for consistent deployment
   - Docker Compose for local development
   - Designed for cloud deployment (AWS/Azure/GCP)

## Features

- **AI-Powered Responses**: Utilizes OpenAI's language models to understand and respond to customer inquiries
- **Knowledge Base Integration**: Semantic search using vector embeddings for retrieving relevant information
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Emotion Analysis**: Detects user sentiment and adapts response style accordingly
- **Confidence Scoring**: Shows confidence levels for AI responses
- **Automated Escalation**: Intelligently escalates low-confidence issues to human support
- **Redis Caching**: Caches high-confidence responses for improved performance
- **Analytics Dashboard**: Provides insights into system usage, performance, and costs
- **Document Management**: Upload and manage knowledge base documents in various formats

## Getting Started

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Python 3.10+ (for local development)
- Node.js 16+ (for local development)

### Deployment Options

#### 1. Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Powered-Enterprise-Customer-Support-System.git
cd AI-Powered-Enterprise-Customer-Support-System

# Create .env file with required variables (see Environment Variables section)
cp .env_example .env
# Edit .env with your configuration

# Start the services
docker-compose up -d

# Initialize the database (first time only)
docker-compose exec backend python init_db.py
```

#### 2. Local Development

See the [Local Development Guide](LOCAL_DEVELOPMENT_GUIDE.md) for detailed instructions.

#### 3. Production Deployment

For production deployment on AWS, see the [AWS Deployment Guide](AWS_DEPLOYMENT_GUIDE.md).

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/customer_support

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Security
JWT_SECRET=your_jwt_secret_key
JWT_EXPIRATION=86400  # 24 hours in seconds

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_CACHE_TTL=86400
```

## Usage

### User Guide

1. **Registration/Login**: Create an account or log in with existing credentials
2. **Chat Interface**: Ask questions related to your products or services
3. **View Responses**: AI will provide responses with confidence scores
4. **Escalation**: Complex issues will be automatically escalated, or you can manually request human support

### Admin Guide

1. **Admin Dashboard**: Access the admin panel via the dashboard button (admins only)
2. **Upload Documents**: Add documents to the knowledge base
3. **View Analytics**: Monitor system usage, performance metrics, and costs
4. **User Management**: Manage user accounts and permissions

## Development

### Code Structure

```
├── backend/                # FastAPI backend
│   ├── main.py             # Main application file
│   ├── models.py           # Database models
│   └── knowledge_base.py   # Vector search implementation
├── frontend/               # React frontend
│   ├── src/                # Source code
│   │   ├── pages/          # Page components
│   │   └── components/     # Reusable components
├── data/                   # Data storage
└── docker-compose.yml      # Docker configuration
```

### Testing

```bash
# Run stress test
python stress_test.py

# Run database performance test
python db_performance_test.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language models
- FAISS for vector search capabilities
- FastAPI for the backend framework
- React for the frontend framework
