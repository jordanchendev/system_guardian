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
# Copy the example environment file and modify with your own values
cp .env.example .env
```

Then update the `.env` file with your configuration:

```bash
SYSTEM_GUARDIAN_OPENAI_API_KEY=your-openai-api-key
SYSTEM_GUARDIAN_SLACK_BOT_TOKEN=your-slack-bot-token
SYSTEM_GUARDIAN_SLACK_CHANNEL_ID=channel-id
SYSTEM_GUARDIAN_JIRA_URL=your_jira_url
SYSTEM_GUARDIAN_JIRA_USERNAME=your_jira_email
SYSTEM_GUARDIAN_JIRA_API_TOKEN=your_jira_api_token
SYSTEM_GUARDIAN_JIRA_PROJECT_KEY=your_project_key
```

### 3️⃣ Run the Application

#### Development Environment

Start the development environment services (Database, RabbitMQ, Qdrant) using Docker Compose:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

Then run the FastAPI service locally:

```bash
poetry run python -m system_guardian
```

#### Docker Compose Setup (Experimental)

> **Note**: The whole project Docker Compose setup is currently under development and not yet ready for use. 

```bash
docker-compose -f docker-compose.yml up --build
```

This will launch the following services:
- API service (system_guardian)
- Event consumer service (event_consumer_service)
- PostgreSQL database
- RabbitMQ message queue
- Qdrant vector database

### 4️⃣ Webhook Setup with ngrok

To receive webhooks from external services (GitHub, Jira, etc.), you'll need to expose your local server to the internet. We recommend using ngrok for this purpose:

1. Install ngrok:
```bash
# macOS with Homebrew
brew install ngrok

# Windows with Chocolatey
choco install ngrok

# Linux
snap install ngrok
```

2. Sign up for a free ngrok account at [https://ngrok.com](https://ngrok.com) and get your authtoken

3. Configure ngrok with your authtoken:
```bash
ngrok config add-authtoken your-authtoken
```

4. Start ngrok to expose your FastAPI server:
```bash
ngrok http --domain=your-domain 5566
```

5. Update your webhook URLs in external services with the ngrok URL:
- GitHub webhook URL: `https://your-ngrok-url/api/v1/ingest/github/`
- Jira webhook URL: `https://your-ngrok-url/api/v1/ingest/jira/`
- Datadog webhook URL: `https://your-ngrok-url/api/v1/ingest/datadog/` 

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

## 📄 License

This project is released under the [MIT License](LICENSE).


