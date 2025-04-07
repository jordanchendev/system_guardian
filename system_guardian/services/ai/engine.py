from typing import List, Dict, Optional, Any, Union, Callable
import json
from datetime import datetime
import time
from functools import lru_cache
import sqlalchemy
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_litellm.chat_models import ChatLiteLLM
from system_guardian.settings import settings

from loguru import logger


class AIEngine:
    """AI engine for generating resolution suggestions and insights."""

    def __init__(
        self,
        vector_db_client,
        llm_client: Optional[ChatLiteLLM] = None,
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-3.5-turbo",
        cache_size: int = 100,
        enable_metrics: bool = True,
        plugins: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the AI Engine.

        :param vector_db_client: Client for vector database operations
        :param llm_client: Optional ChatLiteLLM client for LLM operations
        :param embedding_model: Model to use for embeddings generation
        :param llm_model: Default LLM model to use for text generation
        :param cache_size: Size of the LRU cache for embeddings
        :param enable_metrics: Whether to track performance metrics
        :param plugins: Optional dictionary of plugin instances to extend functionality
        """
        self.vector_db = vector_db_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.cache_size = cache_size
        self.enable_metrics = enable_metrics
        self.metrics = {
            "embedding_calls": 0,
            "embedding_errors": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "vector_search_calls": 0,
            "total_processing_time": 0,
        }

        # Initialize LLM client
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = ChatLiteLLM(
                model=self.llm_model,
                api_key=settings.openai_api_key,
            )

        # Initialize plugins
        self.plugins = plugins or {}

        logger.debug(
            f"Initialized AIEngine with embedding model: {embedding_model}, LLM model: {llm_model}"
        )

        # Configure the embedding cache
        self._configure_embedding_cache()

    def _configure_embedding_cache(self):
        """Configure the LRU cache for embeddings."""

        @lru_cache(maxsize=self.cache_size)
        def _cached_embedding(text: str) -> List[float]:
            # This is just a placeholder to make the cache work
            # The actual implementation will call the async method
            return []

        self._embedding_cache = _cached_embedding
        logger.debug(f"Configured embedding cache with size: {self.cache_size}")

    def _track_metric(self, metric_name: str, increment: int = 1):
        """Track a performance metric if metrics are enabled."""
        if self.enable_metrics:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + increment

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current performance metrics."""
        logger.debug(f"AIEngine metrics: {self.metrics}")
        return self.metrics

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for text.

        :param text: Text to generate embedding for
        :return: Vector embedding as list of floats
        """
        # Ensure text is a string
        if not isinstance(text, str):
            logger.warning(
                f"Input text is not a string but {type(text)}, converting to string"
            )
            text = str(text)

        # Check if we have it in cache (using a cache key)
        cache_key = text.strip()[:1000]  # Limit key size

        # Track call
        self._track_metric("embedding_calls")
        start_time = time.time()

        try:
            # Try to get from LRU cache
            # Since lru_cache doesn't work with async functions directly,
            # we use it as a lookup mechanism only
            cached_result = self._embedding_cache(cache_key)
            if cached_result:
                logger.debug("Embedding cache hit")
                return cached_result

            # Check if llm client is None
            if self.llm is None:
                logger.error("LLM client is None, cannot generate embedding")
                return [0.0] * 1536  # Default embedding size

            # Generate new embedding
            logger.debug(f"Generating embedding using model: {self.embedding_model}")
            response = await self.llm.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding

            # Update cache by calling the function (not ideal but works for this case)
            # In a production system, a proper async cache would be better
            self._embedding_cache.cache_clear()  # Clear to avoid growing too much
            self._embedding_cache(cache_key)
            self._embedding_cache.__wrapped__.__dict__[cache_key] = embedding

            return embedding
        except Exception as e:
            # Log and track the error
            self._track_metric("embedding_errors")
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to zero vector
            return [0.0] * 1536  # Default embedding size
        finally:
            # Track processing time
            processing_time = time.time() - start_time
            self._track_metric("total_processing_time", processing_time)

    async def find_similar_incidents(
        self,
        incident_text: str,
        limit: int = 5,
        filter_condition: Optional[Dict] = None,
        min_similarity_score: float = 0.5,
    ) -> List[Dict]:
        """
        Find similar past incidents using vector similarity.

        :param incident_text: Text to search for similar incidents
        :param limit: Maximum number of results to return
        :param filter_condition: Optional filter condition for the query
        :param min_similarity_score: Minimum similarity score for returned incidents
        :return: List of similar incidents with similarity scores
        """
        start_time = time.time()
        self._track_metric("vector_search_calls")

        try:
            # Ensure incident_text is a string
            if not isinstance(incident_text, str):
                logger.warning(
                    f"incident_text is not a string but {type(incident_text)}, converting to string"
                )
                incident_text = str(incident_text)

            # Generate embedding for the query text
            embedding = await self.generate_embedding(incident_text)

            # Search the vector database
            logger.debug(
                f"Searching vector database with limit: {limit*2}, filter: {filter_condition}"
            )
            results = await self.vector_db.search_vectors(
                collection_name="incident_vectors",
                query_vector=embedding,
                limit=limit * 2,  # Request more than needed to account for filtering
                filter_condition=filter_condition,
            )

            # Filter by similarity score - handle VectorRecord objects properly
            filtered_results = []
            for record in results:
                # Check if we have a VectorRecord object with a score attribute
                if hasattr(record, "score") and record.score is not None:
                    if record.score >= min_similarity_score:
                        # Convert to dictionary format expected by other functions
                        incident_dict = {
                            "incident_id": record.metadata.get("incident_id", ""),
                            "title": record.metadata.get("title", ""),
                            "description": record.metadata.get("description", ""),
                            "severity": record.metadata.get("severity", ""),
                            "status": record.metadata.get("status", ""),
                            "source": record.metadata.get("source", ""),
                            "created_at": record.metadata.get("created_at", ""),
                            "similarity_score": record.score,
                        }
                        filtered_results.append(incident_dict)
                # Fallback for legacy dictionary format
                elif (
                    isinstance(record, dict)
                    and record.get("similarity_score", 0) >= min_similarity_score
                ):
                    filtered_results.append(record)

            logger.info(
                f"Found {len(filtered_results)} similar incidents with similarity score >= {min_similarity_score}"
            )
            # Return up to the requested limit
            return filtered_results[:limit]
        except Exception as e:
            logger.error(f"Error finding similar incidents: {str(e)}", exc_info=True)
            return []
        finally:
            processing_time = time.time() - start_time
            self._track_metric("total_processing_time", processing_time)

    # Helper method to get incident details
    async def _get_incident_details(self, db_session, incident_id: int) -> Dict:
        """
        Get incident details from the database.

        :param db_session: Database session
        :param incident_id: ID of the incident
        :return: Dictionary with incident details or None if not found
        """
        from system_guardian.db.models.incidents import Incident, Event, Resolution

        try:
            # Query the incident
            incident_query = select(Incident).where(Incident.id == incident_id)
            result = await db_session.execute(incident_query)
            incident = result.scalars().first()

            if not incident:
                logger.warning(f"Incident with ID {incident_id} not found")
                return None

            # Get all related events
            events_query = select(Event).where(Event.related_incident_id == incident_id)
            events_result = await db_session.execute(events_query)
            related_events = events_result.scalars().all()
            logger.debug(
                f"Found {len(related_events)} events for incident {incident_id}"
            )

            # Get resolution if any
            resolution_query = select(Resolution).where(
                Resolution.incident_id == incident_id
            )
            resolution_result = await db_session.execute(resolution_query)
            resolution = resolution_result.scalars().first()

            # Format incident data
            incident_data = {
                "id": incident.id,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity,
                "status": incident.status,
                "source": incident.source,
                "created_at": incident.created_at.isoformat(),
                "resolved_at": (
                    incident.resolved_at.isoformat() if incident.resolved_at else None
                ),
                "resolution": resolution.suggestion if resolution else None,
                "similarity_score": 1.0,  # Perfect match with itself
                "events": [
                    {
                        "id": event.id,
                        "source": event.source,
                        "event_type": event.event_type,
                        "content": event.content,
                        "created_at": event.created_at.isoformat(),
                    }
                    for event in related_events
                ],
            }

            return incident_data
        except Exception as e:
            logger.error(f"Error getting incident details: {str(e)}", exc_info=True)
            return None

    # Helper method to generate query text from incident
    async def _generate_query_text_from_incident(self, incident_data: Dict) -> str:
        """
        Generate query text from incident data for vector search.

        :param incident_data: Incident data
        :return: Query text
        """
        # Create a text representation from the incident title and description
        query_text = f"{incident_data.get('title', '')}"
        if incident_data.get("description"):
            query_text += f" {incident_data.get('description', '')}"

        # Add event summaries if available
        for event in incident_data.get("events", [])[:3]:  # Just use first 3 events
            if "summary" in event:
                query_text += f" {event['summary']}"

        # Truncate if too long
        if len(query_text) > 1000:
            query_text = query_text[:1000]

        return query_text

    async def find_related_incidents(
        self,
        db_session,
        incident_id: Optional[int] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        include_resolved: bool = True,
        min_similarity_score: float = 0.5,
    ) -> Dict:
        """
        Find similar past incidents and provide insights based on them.

        :param db_session: Database session
        :param incident_id: Optional ID of the incident to find related incidents for
        :param query_text: Optional text to search for related incidents
        :param limit: Maximum number of related incidents to return
        :param include_resolved: Whether to include resolved incidents
        :param min_similarity_score: Minimum similarity score for related incidents
        :return: Dict with related incidents, insights, and current incident
        """
        from system_guardian.services.ai.incident_similarity import (
            IncidentSimilarityService,
        )

        start_time = time.time()
        logger.info(
            f"Finding related incidents for incident_id={incident_id}, query_text_provided={bool(query_text)}"
        )

        try:
            # Initialize similarity service
            similarity_service = IncidentSimilarityService(
                qdrant_client=self.vector_db,
                openai_client=self.llm,
            )

            incident_data = None
            final_query_text = query_text

            # If incident ID is provided, get incident details
            if incident_id:
                incident_data = await self._get_incident_details(
                    db_session, incident_id
                )

                if not incident_data:
                    raise ValueError(f"Incident with ID {incident_id} not found")

                # Create query text if not provided
                if not final_query_text:
                    final_query_text = await self._generate_query_text_from_incident(
                        incident_data
                    )

            if not final_query_text:
                raise ValueError("Either incident_id or query_text must be provided")

            # Define filter condition based on parameters
            filter_condition = {}
            if not include_resolved:
                filter_condition = {
                    "must": [
                        {"key": "status", "match": {"any": ["open", "investigating"]}}
                    ]
                }
                logger.debug("Applied filter to exclude resolved incidents")

            # Find similar incidents
            logger.debug(f"Searching for similar incidents with limit={limit}")
            similar_incidents_data = await similarity_service.find_similar_incidents(
                query_text=final_query_text,
                limit=limit,
                filter_condition=filter_condition,
            )

            # Filter by minimum similarity score
            similar_incidents_data = [
                incident
                for incident in similar_incidents_data
                if incident.get("similarity_score", 0) >= min_similarity_score
            ]
            logger.info(
                f"Found {len(similar_incidents_data)} similar incidents with score >= {min_similarity_score}"
            )

            # Convert to standardized format
            related_incidents = self._standardize_incident_results(
                similar_incidents_data,
                current_incident_id=incident_data["id"] if incident_data else None,
            )

            # Generate insights based on related incidents
            logger.debug("Generating insights based on related incidents")
            insights = await self.generate_insights(
                current_incident=incident_data, related_incidents=related_incidents
            )

            return {
                "incidents": related_incidents,
                "insights": insights,
                "current_incident": incident_data,
            }
        except Exception as e:
            logger.error(f"Error finding related incidents: {str(e)}", exc_info=True)
            raise
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"find_related_incidents completed in {processing_time:.2f}s")
            self._track_metric("total_processing_time", processing_time)

    def _standardize_incident_results(
        self, incidents: List[Dict], current_incident_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Standardize incident results from the vector database.

        :param incidents: List of incidents from the vector database
        :param current_incident_id: ID of the current incident to exclude
        :return: List of standardized incident dictionaries
        """
        standardized_incidents = []

        for incident in incidents:
            # Skip the current incident if it's in the results
            if current_incident_id and str(current_incident_id) == str(
                incident.get("incident_id")
            ):
                continue

            standardized_incidents.append(
                {
                    "id": (
                        int(incident.get("incident_id"))
                        if incident.get("incident_id")
                        else 0
                    ),
                    "title": incident.get("title", ""),
                    "description": incident.get("description", ""),
                    "severity": incident.get("severity", "medium"),
                    "status": incident.get("status", "unknown"),
                    "source": incident.get("source", "unknown"),
                    "created_at": incident.get(
                        "created_at", datetime.utcnow().isoformat()
                    ),
                    "resolved_at": incident.get("resolved_at"),
                    "resolution": incident.get("resolution"),
                    "similarity_score": incident.get("similarity_score", 0),
                }
            )

        logger.debug(f"Standardized {len(standardized_incidents)} incident results")
        return standardized_incidents

    async def generate_insights(
        self, current_incident: Optional[Dict], related_incidents: List[Dict]
    ) -> List[Dict]:
        """
        Generate insights based on the current incident and related incidents.

        :param current_incident: The current incident data if available
        :param related_incidents: List of related incidents
        :returns: List of insights derived from the incidents
        """
        if not related_incidents:
            logger.info("No related incidents provided, skipping insights generation")
            return []

        start_time = time.time()
        self._track_metric("llm_calls")
        logger.info(
            f"Generating insights based on {len(related_incidents)} related incidents"
        )

        try:
            # Prepare incident data for the prompt
            current_incident_text = self._format_incident_for_prompt(
                current_incident, is_current=True
            )
            related_incidents_text = self._format_related_incidents_for_prompt(
                related_incidents
            )

            # Create prompt for the LLM
            prompt = f"""
            {current_incident_text}
            
            {related_incidents_text}
            
            Based on the information above, generate 3-5 key insights about these incidents. 
            Each insight should be in the following format:
            - type: the type of insight (pattern, frequency, severity, resolution, etc.)
            - description: detailed explanation of the insight
            - confidence: a number between 0 and 1 indicating how confident you are about this insight
            
            Example insight:
            {{
                "type": "pattern",
                "description": "80% of similar incidents originate from the authentication service, suggesting a systemic issue in that component.",
                "confidence": 0.85
            }}
            
            Return the insights as a JSON array with a key called "insights".
            """

            # Generate insights using LLM
            logger.debug(
                f"Calling LLM with model {self.llm_model} to generate insights"
            )
            response = await self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing IT incidents and identifying patterns and insights. Provide your response as a valid JSON object with an 'insights' array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Parse insights from response
            insights_text = response.choices[0].message.content
            insights_data = json.loads(insights_text)

            # Convert to standardized format
            insights = []
            for insight in insights_data.get("insights", []):
                insights.append(
                    {
                        "type": insight.get("type", "unknown"),
                        "description": insight.get("description", ""),
                        "confidence": float(insight.get("confidence", 0.5)),
                    }
                )

            logger.info(f"Generated {len(insights)} insights")
            return insights
        except Exception as e:
            # In case of error, log and return a generic insight
            self._track_metric("llm_errors")
            logger.exception(f"Error generating insights: {str(e)}")
            return [
                {
                    "type": "error",
                    "description": f"Failed to generate insights: {str(e)}",
                    "confidence": 0.1,
                }
            ]
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Insights generation completed in {processing_time:.2f}s")
            self._track_metric("total_processing_time", processing_time)

    def _format_incident_for_prompt(
        self, incident: Optional[Dict], is_current: bool = False
    ) -> str:
        """
        Format incident data for inclusion in an LLM prompt.

        :param incident: Incident data dictionary
        :param is_current: Whether this is the current incident
        :return: Formatted text for the prompt
        """
        if not incident:
            return ""

        header = "Current Incident:" if is_current else "Incident:"

        formatted_text = f"""
        {header}
        ID: {incident["id"]}
        Title: {incident["title"]}
        Description: {incident["description"] or 'N/A'}
        Severity: {incident["severity"]}
        Status: {incident["status"]}
        Source: {incident["source"]}
        Created: {incident["created_at"]}
        """

        if not is_current and incident.get("resolved_at"):
            formatted_text += f"Resolved: {incident['resolved_at']}\n"

        if incident.get("resolution"):
            formatted_text += f"Resolution: {incident['resolution']}\n"

        if not is_current and incident.get("similarity_score") is not None:
            formatted_text += f"Similarity Score: {incident['similarity_score']:.2f}\n"

        return formatted_text

    def _format_related_incidents_for_prompt(self, incidents: List[Dict]) -> str:
        """
        Format a list of related incidents for inclusion in an LLM prompt.

        :param incidents: List of incident dictionaries
        :return: Formatted text for the prompt
        """
        if not incidents:
            return "No related incidents found."

        formatted_text = "Related Incidents:\n"

        for i, incident in enumerate(incidents):
            formatted_text += f"""
            Incident {i+1}:
            ID: {incident["id"]}
            Title: {incident["title"]}
            Description: {incident["description"] or 'N/A'}
            Severity: {incident["severity"]}
            Status: {incident["status"]}
            Source: {incident["source"]}
            Created: {incident["created_at"]}
            Resolved: {incident["resolved_at"] or 'Not resolved'}
            Resolution: {incident["resolution"] or 'No resolution provided'}
            Similarity Score: {incident["similarity_score"]:.2f}
            """

        return formatted_text

    async def generate_resolution(
        self,
        incident_id: int,
        session: AsyncSession,
        model: Optional[str] = None,
        temperature: float = 0.3,
        store_result: bool = True,
    ) -> Dict:
        """
        Generate resolution suggestion for an incident.

        :param incident_id: The ID of the incident to generate a resolution for
        :param session: Database session to use for querying
        :param model: Optional model override for the LLM
        :param temperature: Temperature for the LLM generation (0.0-1.0)
        :param store_result: Whether to store the resolution in the database
        :return: Dictionary with resolution text, confidence, and metadata
        """
        logger.info(f"Generating resolution for incident ID: {incident_id}")

        result = await self.generate_resolution_with_generator(
            incident_id=incident_id,
            session=session,
            force_regenerate=True,  # Always generate a new resolution when this method is called
            model=model,
            temperature=temperature,
        )

        # If store_result is False, we need to handle it here since generate_resolution_with_generator
        # always stores the result by default
        if not store_result and result:
            # Find and delete the resolution that was just created
            from system_guardian.db.models.incidents import Resolution

            try:
                # Get the latest resolution for this incident
                resolution_query = (
                    select(Resolution)
                    .where(Resolution.incident_id == incident_id)
                    .order_by(sqlalchemy.desc(Resolution.generated_at))
                )

                resolution_result = await session.execute(resolution_query)
                resolution = resolution_result.scalars().first()

                if resolution:
                    # Delete it
                    await session.delete(resolution)
                    await session.commit()
                    logger.debug(
                        f"Deleted resolution for incident ID: {incident_id} as store_result=False"
                    )
            except Exception as e:
                logger.error(f"Error handling store_result=False: {str(e)}")
                # Rollback the session
                await session.rollback()

        return result

    async def generate_resolution_with_generator(
        self,
        incident_id: int,
        session: AsyncSession,
        force_regenerate: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a resolution for an incident using ResolutionGenerator.

        :param incident_id: ID of the incident
        :param session: Database session
        :param force_regenerate: Force regeneration even if resolution exists
        :param model: Optional model to use
        :param temperature: Temperature for generation
        :return: Resolution data or None if failed
        """
        # Lazy import to avoid circular imports
        from system_guardian.services.ai.resolution_generator import ResolutionGenerator

        # Create a resolution generator
        resolution_generator = ResolutionGenerator(ai_engine=self)

        # Generate resolution
        return await resolution_generator.generate_resolution(
            incident_id=incident_id,
            session=session,
            force_regenerate=force_regenerate,
            model=model,
            temperature=temperature,
        )

    def register_plugin(self, name: str, plugin_instance: Any) -> None:
        """
        Register a plugin to extend AIEngine functionality.

        :param name: Name of the plugin
        :param plugin_instance: Plugin instance
        """
        self.plugins[name] = plugin_instance
        logger.info(f"Registered plugin: {name}")

    def get_plugin(self, name: str) -> Any:
        """
        Get a registered plugin by name.

        :param name: Name of the plugin to get
        :return: Plugin instance
        :raises: KeyError if plugin not found
        """
        if name not in self.plugins:
            raise KeyError(f"Plugin not found: {name}")
        return self.plugins[name]

    def has_plugin(self, name: str) -> bool:
        """
        Check if a plugin is registered.

        :param name: Name of the plugin to check
        :return: True if plugin exists, False otherwise
        """
        return name in self.plugins
