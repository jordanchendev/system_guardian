# System Guardian 🛡️

An AI-powered incident management platform designed to autonomously monitor, analyze, and suggest resolutions for on-call incidents. System Guardian integrates with tools like Slack, GitHub, Datadog, and more to provide **real-time insights** and **AI-driven remediation suggestions**.

## 🌟 Features

- **Real-time Incident Detection**: Ingests and processes events from **Datadog, GitHub, Jira**, and other sources
- **AI-Powered Resolution Suggestions**: Uses **GPT-4o / GTP-4o-mini** and **retrieval-based search** with **Qdrant** to suggest fixes based on historical incidents
- **Event-Driven Architecture**: Utilizes **RabbitMQ** for reliable message streaming and processing
- **Scalable & Modular**: Microservice-based structure with **FastAPI**, **PostgreSQL**, and **Elasticsearch**
- **Monitoring & Logging**: Tracks incidents, logs, and system health using **Datadog & ELK stack**

---

## 🚀 Getting Started

### 1️⃣ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/your-repo/system_guardian.git
cd system_guardian
python -m venv venv
source venv/bin/activate
pip install poetry
poetry install
```

### 2️⃣ Environment Configuration

Create a `.env` file in the project root directory and configure the necessary settings:

```bash
SYSTEM_GUARDIAN_DB_HOST=localhost
SYSTEM_GUARDIAN_DB_PORT=5432
SYSTEM_GUARDIAN_DB_USER=system_guardian
SYSTEM_GUARDIAN_DB_PASS=system_guardian
SYSTEM_GUARDIAN_DB_BASE=system_guardian
SYSTEM_GUARDIAN_RABBIT_HOST=localhost
OPENAI_API_KEY=your-openai-api-key
SLACK_BOT_TOKEN=your-slack-bot-token
GITHUB_WEBHOOK_SECRET=your-github-webhook-secret
SYSTEM_GUARDIAN_QDRANT_HOST=localhost
SYSTEM_GUARDIAN_QDRANT_PORT=6333
```

### 3️⃣ Run the Application

#### Development Environment

Start the development environment services (Database, RabbitMQ, Qdrant) using Docker Compose:

```bash
docker-compose -f deploy/docker-compose.dev.yml up --build
```

Then run the FastAPI service locally:

```bash
poetry run python -m system_guardian
```

#### Production Environment

Start the complete production environment (including all services) using Docker Compose from the project root directory:

```bash
docker-compose -f deploy/docker-compose.yml up --build
```

This will launch the following services:
- API service (system_guardian)
- Event consumer service (event_consumer_service)
- PostgreSQL database
- RabbitMQ message queue
- Qdrant vector database

---

## 📊 Visualization Interfaces

System Guardian provides several visualization interfaces to help you monitor and manage various components of the system:

### Qdrant Vector Database Dashboard

Qdrant offers an intuitive web interface for managing and monitoring vector collections:

- **Access URL**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Features**:
  - View and manage vector collections
  - Monitor collection statistics
  - Execute vector searches and test queries
  - View collection configurations and index settings

### RabbitMQ Management Interface

The RabbitMQ management interface allows you to monitor and manage the message queuing system:

- **Access URL**: [http://localhost:15672](http://localhost:15672)
- **Default Credentials**:
  - Username: guest
  - Password: guest
- **Features**:
  - Monitor queue status and message traffic
  - View exchanges and binding configurations
  - Publish and receive test messages
  - Manage users and permissions
  - View performance metrics and system resource usage

### FastAPI Swagger Documentation

FastAPI's auto-generated API documentation interface:

- **Access URL**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

These visualization interfaces greatly simplify the development and debugging process, allowing you to intuitively understand the operational status of the system.

---

## 📝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) to learn how to participate in this project.

## 📄 License

This project is released under the [MIT License](LICENSE).


