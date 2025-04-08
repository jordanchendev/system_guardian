from typing import List, Dict, Optional, Any, Union, Callable
import json
from datetime import datetime
import time
from functools import lru_cache
import sqlalchemy
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_litellm.chat_models import ChatLiteLLM
from litellm import embedding
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

            # Generate new embedding using LiteLLM
            logger.debug(f"Generating embedding using model: {self.embedding_model}")
            response = embedding(
                model=self.embedding_model,
                input=[text],
                api_key=settings.openai_api_key,
            )
            embedding_vector = response.data[0]["embedding"]

            # Update cache by calling the function (not ideal but works for this case)
            # In a production system, a proper async cache would be better
            self._embedding_cache.cache_clear()  # Clear to avoid growing too much
            self._embedding_cache(cache_key)
            self._embedding_cache.__wrapped__.__dict__[cache_key] = embedding_vector

            return embedding_vector
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
        Find similar past incidents using vector similarity and LLM relevance judgment.

        :param incident_text: Text to search for similar incidents
        :param limit: Maximum number of results to return
        :param filter_condition: Optional filter condition for the query
        :param min_similarity_score: Minimum similarity score for initial filtering
        :return: List of similar incidents with relevance scores
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

            # Search the vector database with a larger limit to allow for LLM filtering
            logger.debug(
                f"Searching vector database with limit: {limit*2}, filter: {filter_condition}"
            )
            results = await self.vector_db.search_vectors(
                collection_name="incident_vectors",
                query_vector=embedding,
                limit=limit
                * 2,  # Request more than needed to account for LLM filtering
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

            if not filtered_results:
                logger.info("No incidents found above similarity threshold")
                return []

            # Use LLM to judge relevance and rank incidents
            relevance_prompt = f"""
            You are an expert at analyzing IT incidents and determining their relevance to each other.
            
            Current Incident:
            {incident_text}
            
            Please analyze the following incidents and determine how relevant they are to the current incident.
            Consider:
            1. Technical similarity (same components, error types, etc.)
            2. Root cause similarity
            3. Impact similarity
            4. Resolution approach similarity
            
            For each incident, provide:
            - relevance_score: A number between 0 and 1 indicating how relevant it is
            - relevance_reason: A brief explanation of why it's relevant
            
            Incidents to analyze:
            {json.dumps(filtered_results, indent=2)}
            
            Return a JSON array with the same incidents, but with added relevance_score and relevance_reason fields.
            """

            # Call LLM for relevance judgment
            response = await self.llm.ainvoke(
                [
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing IT incidents and determining their relevance to each other. Provide your response as a valid JSON array.",
                    },
                    {"role": "user", "content": relevance_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Parse and process LLM response
            try:
                # Extract content from response
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Try to find JSON content in the response
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    relevance_data = json.loads(json_str)
                else:
                    raise json.JSONDecodeError(
                        "No JSON object found in response", response_content, 0
                    )

                # Extract incidents array from response
                if isinstance(relevance_data, dict):
                    incidents = relevance_data.get("incidents", [])
                else:
                    incidents = relevance_data

                if not isinstance(incidents, list):
                    logger.error(
                        f"Unexpected response format. Expected list but got {type(incidents)}"
                    )
                    return filtered_results[:limit]

                # Sort by relevance score and take top results
                sorted_results = sorted(
                    incidents,
                    key=lambda x: float(x.get("relevance_score", 0)),
                    reverse=True,
                )[:limit]

                logger.info(
                    f"Found {len(sorted_results)} relevant incidents after LLM analysis"
                )
                return sorted_results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM relevance response: {str(e)}")
                # Fall back to similarity-based results
                return filtered_results[:limit]
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Error finding similar incidents: {str(e)}", exc_info=True)
            return []
        finally:
            processing_time = time.time() - start_time
            self._track_metric("total_processing_time", processing_time)

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
            response = await self.llm.ainvoke(
                [
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
            insights_text = response.content
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
