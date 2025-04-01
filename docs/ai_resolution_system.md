# System Guardian Technical Architecture Document

## Overview

System Guardian is an AI-driven incident management platform designed to autonomously monitor, analyze, and provide solution recommendations for on-call incidents. This platform integrates with Slack, GitHub, Jira, Datadog, and other tools to provide real-time insights and AI-driven remediation suggestions.

## Technical Architecture

### Data Extraction and Processing Flow

System Guardian uses an event-driven microservices architecture to extract event data from multiple sources:

1. **Multi-source Data Extraction**
   - **GitHub Integration**: Monitor code commits, issues, and deployment events
   - **Jira Integration**: Track issues and comments
   - **Datadog Integration**: Handle system alerts and monitoring data
   - **Technical Documentation Integration**: Support uploading technical documentation to Qdrant vector database as a knowledge base reference for AI solution generation, improving the accuracy and reliability of AI solutions

2. **Uniform Message Specification**
   - Implement a uniform event model (Event model), converting data from different sources into standard format
   - Use JSONB data type to save complete original event content, ensuring context completeness
   - Implement cross-platform event classification through standardized source, event_type fields
   - Internal message conversion layer ensures consistent processing of external events

3. **Event Processing Flow**
   - Use RabbitMQ as a message queue to ensure event processing reliability
   - Event consumption service (EventConsumerService) is responsible for extracting and processing events from the queue
   - Standardize event data and store it in the PostgreSQL database

4. **Event Analysis and Response Mechanism**
   - System automatically analyzes event data to identify potential incidents
   - Once an incident is confirmed, it immediately sends a notification to the relevant team via Slack robot
   - At the same time, it automatically creates a Jira ticket to track resolution progress
   - Based on incident context and historical data, it generates preliminary resolution suggestion (Resolution) immediately
   - Track notification and response status throughout the process for subsequent optimization

5. **Data Standardization**
   - Convert data from different sources into a unified event and incident model
   - Establish the relationship between events and incidents to build a complete incident context

### AI-driven Analysis and Insights

1. **AI Engine Architecture**
   - Core AIEngine class provides vector embedding, text generation, and similarity search functionality
   - Supports multiple LLM models (GPT-4o / GPT-4o mini), and can switch based on demand
   - Use OpenAI Embeddings API to generate vector representations

2. **Vector Database**
   - Utilizes Qdrant vector database for efficient storage and retrieval of event and knowledge embeddings
   - Enables high-performance similarity search for finding related incidents and documents

3. **AI Analysis Service**
   - **Incident Detector (IncidentDetector)**: Automatically identify potential events
   - **Incident Analyzer (IncidentAnalyzer)**: Deeply analyze event causes and impacts
   - **Severity Classifier (SeverityClassifier)**: Evaluate event severity
   - **Similarity Engine (IncidentSimilarity)**: Identify related historical events

4. **Solution Generation**
   - **Solution Generator (ResolutionGenerator)**: Based on historical data, technical documentation, and context to generate remediation suggestions
   - **Knowledge Retrieval (KnowledgeRetrieval)**: Query related technical documentation fragments from vector database as reference for solution generation
   - **Report Generator (ReportGenerator)**: Automatically generate event reports and summaries, and reference related technical documentation information

### API Endpoints and Services

System Guardian provides a series of REST API endpoints for integration with other systems:

1. **Data Extraction API**
   - `/api/ingest/github/`: Receive GitHub events
   - `/api/ingest/datadog/`: Handle Datadog alerts
   - `/api/ingest/jira/`: Receive Jira events
   - `/api/vector-db/knowledge/upload`: Upload technical documentation to vector database as knowledge base

2. **Solution and Event Management API**
   - `/api/ai/generate-resolution`: Use AI to generate event resolution suggestion
   - `/api/ai/resolutions/{resolution_id}/apply`: Mark solution as applied
   - `/api/ai/resolutions/{resolution_id}/feedback`: Provide solution feedback
   - `/api/ai/resolutions/incident/{incident_id}`: Get all solutions for a specific event
   - `/api/ai/related-incidents`: Find similar historical events and provide insights

3. **Report and Analysis API**
   - `/api/ai/generate-incident-report`: Generate detailed event report
   - `/api/ai/generate-summary-report`: Generate summary report for a time period
   - `/api/ai/generate-recommendations`: Generate operational suggestions based on event history
   - `/api/ai/metrics`: Get AI engine performance metrics

4. **Advanced Analysis API**
   - `/api/ai/analytics/resolution-time/{time_range}`: Analyze resolution time statistics
   - `/api/ai/analytics/common-failures`: Identify common failure patterns
   - `/api/ai/analytics/ai-effectiveness`: Evaluate AI suggestion effectiveness
   - `/api/ai/analytics/trend-report`: Generate trend analysis report
   - `/api/ai/analytics/root-cause-analysis`: Perform deep root cause analysis


## Data Model

System Guardian uses SQLAlchemy ORM to define the following core models:

1. **Incident**
   - Represents system issues, including title, description, severity, and status
   - Establish associations with triggered events and related events

2. **Event**
   - Event data from various sources
   - Contains source, event type, and complete event content
   - Uniform event model ensures consistent processing of data from different sources

3. **Resolution**
   - AI-generated or manually provided event resolution
   - Contains suggestion text, confidence score, and user feedback

## Scalability and Reliability Design

1. **Scalable Service Architecture**
   - Microservices design allows independent expansion of each component
   - Connection pool configuration optimization ensures high concurrency processing capability

2. **Reliability Measures**
   - RabbitMQ message queue ensures event not lost
   - Sound error handling and retry mechanism
   - Database connection pre-check and recycling strategy

3. **Monitoring and Logging**
   - Comprehensive logging using Loguru
   - AI engine performance metric tracking
   - Database query monitoring

## Technical Decisions, Assumptions and Trade-offs

1. **Model Selection Considerations**
   - **Short-term**: GPT-4o/GPT-4o mini chosen for their strong context understanding capabilities and API availability
   - **Long-term**: Plan to evaluate fine-tuned models on incident-specific data to improve accuracy and reduce costs
   - **Trade-off**: Balance between model quality and inference cost - GPT-4o for critical incidents, GPT-4o mini for routine cases

2. **Vector Database Implementation**
   - **Decision**: Selected Qdrant for its lightweight nature and fast performance
   - **Rationale**: Qdrant offers a lightweight architecture, quick response times, and open-source deployment options
   - **Trade-off**: Accepted increased operational complexity for improved query performance

3. **Data Retention Strategy**
   - **Short-term**: Store complete event data for comprehensive context
   - **Long-term**: Implement intelligent data archiving based on incident relevance
   - **Assumption**: Complete event data is crucial for high-quality AI resolution generation
   - **Trade-off**: Higher storage costs for improved resolution quality

4. **Real-world Deployment Considerations**
   - **Assumption**: API rate limits may impact real-time processing during incident spikes
   - **Mitigation**: Implement priority queuing and intelligent batching for LLM requests
   - **Trade-off**: Accepted potential slight delays for non-critical incidents to ensure system stability

## AI Effectiveness Measurement

1. **Resolution Quality Metrics**
   - **Resolution Adoption Rate**: Percentage of AI suggestions implemented by engineers
   - **Time-to-Resolution Impact**: Comparing resolution time with and without AI assistance
   - **Feedback Scoring System**: Engineers rate suggestions on relevance and applicability

2. **Continuous Improvement Loop**
   - Engineer feedback directly influences model training and resolution generation for future related incidents
   - Regular model evaluation against new incident types to ensure broad coverage
   - Periodic analysis of performance metrics to identify and address improvement opportunities

## Conclusion

System Guardian is a comprehensive AI-driven incident management platform that integrates multi-source data, AI analysis, and solution generation. Its event-driven architecture, uniform message specification, vector search capabilities, and advanced LLM integration enable it to effectively manage and resolve on-call incidents, reduce average resolution time, and improve operational efficiency.

This system is particularly good at learning from historical events to provide relevant solution suggestions for current events and continuously optimizing its suggestion quality based on the results of event resolution. The most important thing is that the implementation of uniform message specification ensures consistent processing and analysis of data from different sources in a single system, greatly improving cross-platform event association and processing efficiency.

## Future Work

System Guardian team is planning the following feature enhancements to further enhance system capabilities:

1. **Solution Confidence Score Redesign**
   - The current confidence score calculation mechanism has inaccuracy issues
   - Plan to redesign the algorithm, combining multiple indicators such as historical success rate, context similarity, and knowledge base reference degree
   - Introduce machine learning models to dynamically adjust scores based on user feedback for continuous optimization

2. **Smart Chatbot**
   - Develop a chat interface supporting natural language interaction, allowing users to query and manage incidents through conversation
   - Implement automatic SQL query capabilities, enabling the chatbot to automatically construct and execute database queries based on user questions
   - Provide incident overview, solution suggestions, and historical incident analysis functions to reduce operational complexity
   - Support cross-platform integration, expanding beyond Slack to communication platforms such as Telegram and Discord
