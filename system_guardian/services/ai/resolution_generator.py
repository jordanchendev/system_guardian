"""
Resolution Generator Module

This module provides functionality to generate and suggest resolutions for incidents.
It uses AI to analyze incident data, related events, and similar past incidents to
generate actionable resolution steps.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from system_guardian.db.models.incidents import Incident, Event, Resolution
from system_guardian.services.ai.service_base import AIServiceBase


class ResolutionGenerator(AIServiceBase):
    """
    Service for generating resolution suggestions for incidents.

    This service uses the AIEngine to generate high-quality resolution
    suggestions based on incident details and similar past incidents.
    """

    # Name of the knowledge collection
    KNOWLEDGE_COLLECTION_NAME = "system_knowledge"

    def __init__(
        self,
        ai_engine,
        service_name: str = "resolution_generator",
        min_confidence_threshold: float = 0.6,
        enable_metrics: bool = True,
    ):
        """
        Initialize the resolution generator service.

        :param ai_engine: AIEngine instance
        :param service_name: Name of the service
        :param min_confidence_threshold: Minimum confidence threshold for generated resolutions
        :param enable_metrics: Whether to track performance metrics
        """
        super().__init__(ai_engine, service_name, enable_metrics)
        self.min_confidence_threshold = min_confidence_threshold

        # Additional metrics specific to resolution generator
        self.metrics.update(
            {
                "resolutions_generated": 0,
                "high_confidence_resolutions": 0,
                "low_confidence_resolutions": 0,
                "knowledge_items_used": 0,
            }
        )

        logger.info(
            f"Initialized ResolutionGenerator with confidence threshold: {min_confidence_threshold}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the service.

        :return: True if initialization was successful, False otherwise
        """
        # No special initialization needed for ResolutionGenerator
        return True

    async def generate_resolution(
        self,
        incident_id: int,
        session: AsyncSession,
        force_regenerate: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a resolution suggestion for an incident.

        :param incident_id: ID of the incident
        :param session: Database session
        :param force_regenerate: Force regeneration even if resolution exists
        :param model: Optional model to use
        :param temperature: Temperature for generation
        :return: Resolution data or None if failed
        """

        async def _generate_resolution():
            from system_guardian.db.models.incidents import Incident, Resolution

            logger.info(f"Generating resolution for incident ID: {incident_id}")

            # Check if resolution already exists (unless force_regenerate)
            if not force_regenerate:
                existing_query = select(Resolution).where(
                    Resolution.incident_id == incident_id
                )
                existing_result = await session.execute(existing_query)
                existing_resolution = existing_result.scalars().first()

                if existing_resolution:
                    logger.info(f"Resolution already exists for incident {incident_id}")
                    return {
                        "resolution_id": existing_resolution.id,
                        "resolution_text": existing_resolution.suggestion,
                        "confidence": existing_resolution.confidence,
                        "incident_id": incident_id,
                        "generated_at": existing_resolution.generated_at.isoformat(),
                        "is_reused": True,
                    }

            # Get incident details
            incident_query = select(Incident).where(Incident.id == incident_id)
            result = await session.execute(incident_query)
            incident = result.scalars().first()

            if not incident:
                logger.error(f"Incident with ID {incident_id} not found")
                return None

            # Perform resolution generation
            try:
                # Generate incident text from incident data
                incident_text = self._format_incident_text(incident)

                # Find similar resolved incidents
                similar_incidents = await self._find_similar_incidents(incident_text)

                # Generate resolution using the AI engine's LLM
                resolution_text, confidence = await self._generate_resolution_text(
                    incident=incident,
                    similar_incidents=similar_incidents,
                    model=model,
                    temperature=temperature,
                )

                # Track confidence metrics
                if confidence >= self.min_confidence_threshold:
                    self._track_metric("high_confidence_resolutions")
                else:
                    self._track_metric("low_confidence_resolutions")

                # Store the resolution
                resolution = Resolution(
                    incident_id=incident_id,
                    suggestion=resolution_text,
                    confidence=confidence,
                    is_applied=False,
                    generated_at=datetime.utcnow(),
                )

                session.add(resolution)
                await session.commit()

                self._track_metric("resolutions_generated")
                logger.info(
                    f"Generated resolution for incident {incident_id} with confidence {confidence:.2f}"
                )

                return {
                    "resolution_id": resolution.id,
                    "resolution_text": resolution_text,
                    "confidence": confidence,
                    "incident_id": incident_id,
                    "generated_at": resolution.generated_at.isoformat(),
                    "model_used": model or self.ai_engine.llm_model,
                    "similar_incidents_count": len(similar_incidents),
                    "is_reused": False,
                }
            except Exception as e:
                await session.rollback()
                logger.error(f"Error generating resolution: {str(e)}")
                raise

        # Use the run_with_metrics helper to track performance
        return await self.run_with_metrics(_generate_resolution)

    async def _find_similar_incidents(self, incident_text: str) -> List[Dict]:
        """
        Find similar incidents using the AI engine.

        :param incident_text: Text representation of the incident
        :returns: List of similar incidents with similarity scores
        """
        try:
            similar_incidents = await self.ai_engine.find_similar_incidents(
                incident_text=incident_text,
                limit=3,
                filter_condition={
                    "must": [{"key": "status", "match": {"any": ["resolved"]}}]
                },
                min_similarity_score=0.6,
            )
            return similar_incidents
        except Exception as e:
            logger.error(f"Error finding similar incidents: {str(e)}")
            return []

    async def _find_relevant_knowledge(
        self, incident_text: str, limit: int = 5
    ) -> List[str]:
        """
        Find relevant knowledge from the knowledge base related to the incident.

        :param incident_text: Text description of the incident
        :param limit: Maximum number of results to return
        :return: List of relevant knowledge items
        """
        try:
            # Generate embedding vector for the query
            embedding = await self.ai_engine.generate_embedding(incident_text)

            # Search for relevant knowledge in the knowledge base
            vector_db = self.ai_engine.vector_db
            knowledge_results = await vector_db.search_vectors(
                collection_name=self.KNOWLEDGE_COLLECTION_NAME,
                query_vector=embedding,
                limit=limit,
            )

            # Extract knowledge text from results
            knowledge_items = []
            for result in knowledge_results:
                if hasattr(result, "metadata") and "text" in result.metadata:
                    knowledge_items.append(result.metadata["text"])

            # Update metrics
            self._track_metric("knowledge_items_used", len(knowledge_items))
            logger.info(f"Found {len(knowledge_items)} relevant knowledge items")

            return knowledge_items
        except Exception as e:
            logger.error(f"Error finding relevant knowledge: {str(e)}")
            return []

    def _format_incident_text(self, incident) -> str:
        """
        Format incident into a text representation.

        :param incident: Incident database model
        :return: Text representation
        """
        incident_text = f"Incident: {incident.title}\n"
        incident_text += f"Description: {incident.description}\n"
        incident_text += f"Severity: {incident.severity}\n"
        incident_text += f"Source: {incident.source}\n"
        incident_text += f"Status: {incident.status}\n"
        incident_text += f"Created at: {incident.created_at}\n"

        return incident_text

    async def _generate_resolution_text(
        self,
        incident,
        similar_incidents: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> tuple:
        """
        Generate resolution text using the AI engine's LLM.

        :param incident: Incident database model
        :param similar_incidents: List of similar incidents
        :param model: Optional model to use
        :param temperature: Temperature for generation
        :return: Tuple of (resolution text, confidence score)
        """
        # Format prompt components
        incident_text = self._format_incident_text(incident)
        similar_incidents_text = self._format_similar_incidents_text(similar_incidents)

        # Get relevant knowledge
        relevant_knowledge = await self._find_relevant_knowledge(incident_text)
        knowledge_text = self._format_knowledge_text(relevant_knowledge)

        # Create the prompt with knowledge
        prompt = self._create_resolution_prompt(
            incident_text, similar_incidents_text, knowledge_text
        )

        # Use the specified model or fall back to default
        model_to_use = model or self.ai_engine.llm_model

        # Call the LLM
        response = await self.ai_engine.llm.chat.completions.create(
            model=model_to_use,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert IT incident resolver. Provide concise, actionable resolution steps.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )

        resolution_text = response.choices[0].message.content

        # Calculate confidence score
        confidence = self._calculate_confidence(
            similar_incidents=similar_incidents,
            incident=incident,
            model=model_to_use,
            knowledge_count=len(relevant_knowledge),
        )

        return resolution_text, confidence

    def _format_similar_incidents_text(self, incidents: List[Dict]) -> str:
        """
        Format similar incidents for the resolution prompt.

        :param incidents: List of similar incidents
        :return: Formatted text
        """
        similar_incidents_text = ""
        for i, similar in enumerate(incidents):
            similar_incidents_text += (
                f"Similar incident {i+1}: {similar.get('title')}\n"
            )
            similar_incidents_text += f"Description: {similar.get('description')}\n"
            if similar.get("resolution"):
                similar_incidents_text += f"Resolution: {similar.get('resolution')}\n"
            similar_incidents_text += (
                f"Similarity score: {similar.get('similarity_score', 0):.2f}\n\n"
            )

        return similar_incidents_text

    def _format_knowledge_text(self, knowledge_items: List[str]) -> str:
        """
        Format relevant knowledge into text format.

        :param knowledge_items: List of knowledge items
        :return: Formatted knowledge text
        """
        if not knowledge_items:
            return "No relevant knowledge found.\n"

        knowledge_text = "## Relevant Knowledge:\n\n"
        for i, item in enumerate(knowledge_items, 1):
            knowledge_text += f"{i}. {item}\n\n"

        return knowledge_text

    def _create_resolution_prompt(
        self, incident_text: str, similar_incidents_text: str, knowledge_text: str = ""
    ) -> str:
        """
        Create prompt for resolution generation.

        :param incident_text: Text representation of incident
        :param similar_incidents_text: Text representation of similar incidents
        :param knowledge_text: Text representation of relevant knowledge
        :return: Complete prompt
        """
        prompt = f"""You are tasked with generating a resolution for the following IT incident:

{incident_text}

"""

        # Add relevant knowledge information
        if knowledge_text:
            prompt += f"""
Here is relevant knowledge that might help with resolving this incident:

{knowledge_text}
"""

        # Add similar incidents information
        if similar_incidents_text:
            prompt += f"""
Here are similar incidents that were resolved in the past:

{similar_incidents_text}
"""

        prompt += """
Based on the incident details, relevant knowledge, and similar past incidents, please provide:

1. A concise resolution suggestion with clear, actionable steps
2. Root cause analysis (if possible)
3. Any preventive measures to avoid similar incidents in the future

Format your response as follows:

## Resolution Steps:
1. [First step]
2. [Second step]
...

## Root Cause:
[Brief root cause analysis]

## Prevention:
[Preventive measures]
"""
        return prompt

    def _calculate_confidence(
        self,
        similar_incidents: List[Dict],
        incident,
        model: str,
        knowledge_count: int = 0,
    ) -> float:
        """
        Calculate confidence score for the resolution.

        :param similar_incidents: List of similar incidents
        :param incident: Incident object
        :param model: Model used for generation
        :param knowledge_count: Number of knowledge items used
        :return: Confidence score (0.0 to 1.0)
        """
        # Base confidence from similar incidents
        base_confidence = 0.5

        # Add confidence based on similar incidents
        if similar_incidents:
            # Average similarity score of past incidents
            avg_similarity = sum(
                inc.get("similarity_score", 0) for inc in similar_incidents
            ) / len(similar_incidents)
            similar_incidents_factor = avg_similarity * 0.3  # 30% weight
        else:
            similar_incidents_factor = 0

        # Add confidence based on knowledge base
        if knowledge_count > 0:
            # More knowledge items means higher confidence
            knowledge_factor = min(0.2, knowledge_count * 0.04)  # 20% weight max
        else:
            knowledge_factor = 0

        # Add confidence based on model
        model_factor = (
            0.1 if "gpt-4" in model.lower() else 0.05
        )  # Better models get higher confidence

        # Calculate total confidence
        confidence = (
            base_confidence + similar_incidents_factor + knowledge_factor + model_factor
        )

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
