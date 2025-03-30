"""Service for classifying incident severity using AI."""
from typing import Dict, Any, Optional, List
import json

from openai import AsyncOpenAI
from loguru import logger

from system_guardian.db.models.incidents import Incident
from system_guardian.settings import settings

class SeverityClassifier:
    """Service for classifying incident severity using AI."""
    
    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-3.5-turbo",
        ai_engine = None,
    ):
        """
        Initialize the severity classifier service.
        
        :param openai_client: OpenAI client instance
        :param model: OpenAI model to use for classification
        :param ai_engine: Optional AIEngine instance for enhanced classification
        """
        self.openai_client = openai_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.severity_levels = ["low", "medium", "high", "critical"]
        self.ai_engine = ai_engine
    
    async def classify_severity(
        self, 
        incident_title: str, 
        incident_description: Optional[str] = None,
        source: Optional[str] = None,
        events_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Classify the severity of an incident based on its content.
        
        :param incident_title: The title of the incident
        :param incident_description: The description of the incident (if available)
        :param source: The source of the incident (e.g., 'github', 'jira')
        :param events_data: Additional event data related to the incident
        :returns: Predicted severity level ('low', 'medium', 'high', 'critical')
        """
        # Prepare the prompt with incident information
        prompt = self._build_classification_prompt(
            title=incident_title,
            description=incident_description,
            source=source,
            events_data=events_data
        )
        
        try:
            # If we have an AIEngine available, use it for classification
            if self.ai_engine:
                logger.debug(f"Using AIEngine for severity classification with model: {self.ai_engine.llm_model}")
                
                response = await self.ai_engine.llm.chat.completions.create(
                    model=self.ai_engine.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an incident severity classifier. "
                                                    "Analyze the incident details and classify its severity "
                                                    "as one of: low, medium, high, critical."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=50     # We only need a short response
                )
                
                # Extract and validate severity from response
                severity = self._extract_severity_from_response(response.choices[0].message.content)
                logger.info(f"AIEngine classified incident severity as {severity}")
                return severity
            
            # Fall back to standard OpenAI client if no AIEngine
            logger.debug(f"Using standard OpenAI client for severity classification with model: {self.model}")
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an incident severity classifier. "
                                                 "Analyze the incident details and classify its severity "
                                                 "as one of: low, medium, high, critical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=50     # We only need a short response
            )
            
            # Extract and validate severity from response
            severity = self._extract_severity_from_response(response.choices[0].message.content)
            logger.info(f"AI classified incident severity as {severity}")
            return severity
            
        except Exception as e:
            logger.error(f"Error classifying incident severity: {str(e)}")
            # Default to 'medium' if classification fails
            return "medium"
    
    def _build_classification_prompt(
        self, 
        title: str, 
        description: Optional[str] = None,
        source: Optional[str] = None,
        events_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build the prompt for severity classification.
        
        :param title: Incident title
        :param description: Incident description
        :param source: Incident source
        :param events_data: Related event data
        :returns: Formatted prompt string
        """
        prompt = f"Incident Title: {title}\n\n"
        
        if description:
            prompt += f"Description: {description}\n\n"
        
        if source:
            prompt += f"Source: {source}\n\n"
        
        if events_data and len(events_data) > 0:
            prompt += "Related Events:\n"
            for event in events_data[:3]:  # Limit to first 3 events to keep prompt size reasonable
                event_str = json.dumps(event, indent=2)
                prompt += f"{event_str}\n\n"
        
        prompt += "\nBased on the above information, classify the severity of this incident as one of: low, medium, high, critical.\n"
        prompt += "Consider factors such as potential business impact, number of affected users, and system criticality."
        
        return prompt
    
    def _extract_severity_from_response(self, response_text: str) -> str:
        """
        Extract the severity level from the AI response.
        
        :param response_text: Text response from OpenAI
        :returns: Validated severity level
        """
        response_text = response_text.lower().strip()
        
        # Check if any severity level is explicitly mentioned
        for level in self.severity_levels:
            if level in response_text:
                return level
        
        # If no exact match, try to find the closest match
        if "critic" in response_text:
            return "critical"
        elif "high" in response_text:
            return "high"
        elif "medium" in response_text or "mod" in response_text:
            return "medium"
        elif "low" in response_text:
            return "low"
        
        # Default to medium if we can't determine severity
        return "medium" 