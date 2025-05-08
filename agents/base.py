from environments.conclave_env import ConclaveEnv
import json
import boto3
from typing import Dict, List, Optional
import logging
import time
import botocore.exceptions

max_tokens = 1000
temperature = 0.5
bedrock_model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

class Agent:
    def __init__(self, agent_id: int, name: str, background: str, env: ConclaveEnv):
        self.agent_id = agent_id
        self.name = name
        self.background = background
        self.bedrock = boto3.client('bedrock-runtime', region_name="us-west-2")
        self.env = env
        self.vote_history = []
        self.logger = logging.getLogger(name)

    def cast_vote(self) -> None:
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history()
        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

Use your vote tool to cast your vote for one of the candidates.
        """
        if (self.agent_id == 0):
            print(prompt)
        self.logger.info(prompt)
        # Define vote tool
        tools = [
            {
                "name": "vote",
                "description": "Cast your vote for one agent (cannot be yourself)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Explain why you chose this agent"
                        },
                        "agent_id": {
                            "type": "integer",
                            "description": "The ID of the agent you're voting for"
                        }
                    },
                    "required": ["agent_id", "reasoning"]
                }
            }
        ]
        try:
            response = self._invoke_claude(prompt, tools)

            # Handle tool call response
            if 'content' in response and len(response['content']) > 0:
                content_item = response['content'][0]

                if content_item['type'] == 'tool_use' and content_item['name'] == 'vote':
                    tool_input = content_item['input']
                    vote = tool_input.get("agent_id")
                    reasoning = tool_input.get("reasoning", "")

                    # Save vote reasoning
                    self.vote_history.append({
                        "vote": vote,
                        "reasoning": reasoning
                    })

                    if vote is not None and isinstance(vote, int):
                        self.env.cast_vote(vote)
                        self.logger.info(f"{self.name} ({self.agent_id}) voted for {self.env.agents[vote].name} ({vote}) because\n{reasoning}")
                        return
                    else:
                        raise ValueError("Invalid vote")

        except Exception as e:
            # Default vote if there's an error
            self.logger.error(f"Error in LlmAgent {self.agent_id} voting: {e}")

    def speaking_urgency(self) -> Dict[str, any]:
        """
        Calculate how urgently the agent wants to speak in the next discussion round.

        Returns:
            Dict with urgency_score (1-100) and reasoning
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history()

        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

Based on the current state of the conclave, how urgently do you feel the need to speak?
Evaluate your desire to speak on a scale from 1-100, where:
1 = You have nothing important to add at this time
100 = You have an extremely urgent point that must be heard immediately

Consider factors such as:
- How strongly do you feel about supporting or opposing specific candidates?
- Do you need to respond to something said in a previous discussion?
- Do you have important information or perspectives that haven't been shared yet?
- Are the voting trends concerning to you?

Use your evaluate_speaking_urgency tool to provide your urgency score and reasoning.
        """
        if (self.agent_id == 0):
            print(prompt)
        self.logger.info(prompt)

        # Define urgency evaluation tool
        tools = [
            {
                "name": "evaluate_speaking_urgency",
                "description": "Evaluate how urgently you want to speak",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "urgency_score": {
                            "type": "integer",
                            "description": "Your urgency score (1-100)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explain why you rated your urgency at this level"
                        }
                    },
                    "required": ["urgency_score", "reasoning"]
                }
            }
        ]

        try:
            response = self._invoke_claude(prompt, tools)

            # Handle tool call response
            if 'content' in response and len(response['content']) > 0:
                content_item = response['content'][0]

                if content_item['type'] == 'tool_use' and content_item['name'] == 'evaluate_speaking_urgency':
                    tool_input = content_item['input']
                    urgency_score = tool_input.get("urgency_score", 50)  # Default to 50 if missing
                    reasoning = tool_input.get("reasoning", "")

                    # Ensure score is in range 1-100
                    urgency_score = max(1, min(100, urgency_score))

                    result = {
                        "agent_id": self.agent_id,
                        "urgency_score": urgency_score,
                        "reasoning": reasoning
                    }

                    self.logger.info(f"{self.name} ({self.agent_id}) speaking urgency: {urgency_score}/100\nReasoning: {reasoning}")
                    return result

                else:
                    raise ValueError("Invalid tool use")

        except Exception as e:
            # Default urgency if there's an error
            self.logger.error(f"Error in LlmAgent {self.agent_id} speaking urgency evaluation: {e}")
            return {
                "agent_id": self.agent_id,
                "urgency_score": 50,  # Default middle urgency
                "reasoning": "Error evaluating speaking urgency"
            }

    def discuss(self, urgency_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate a discussion contribution about the conclave proceedings.

        Args:
            urgency_data: Optional dictionary containing urgency score and reasoning

        Returns:
            Dict with agent_id and message if successful, None otherwise
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history()

        # Include speaking urgency information if available
        urgency_context = ""
        if urgency_data and 'urgency_score' in urgency_data and 'reasoning' in urgency_data:
            urgency_context = f"""You indicated that you have an urgency level of {urgency_data['urgency_score']}/100 to speak.
Your reasoning was: {urgency_data['reasoning']}

Keep this urgency level and reasoning in mind as you formulate your response.
"""

        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

{urgency_context}
It's time for a discussion round. Use your speak tool to contribute to the discussion.
Your goal is to influence others based on your beliefs and background. You can:
1. Make your case for a particular candidate
2. Question the qualifications of other candidates
3. Respond to previous speakers
4. Share your perspectives on what the Church needs

Be authentic to your character and background in your speech.
        """
        if (self.agent_id == 0):
            print(prompt)
        self.logger.info(prompt)

        # Define speak tool
        tools = [
            {
                "name": "speak",
                "description": "Contribute to the conclave discussion",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Your contribution to the discussion (200-500 words)"
                        }
                    },
                    "required": ["message"]
                }
            }
        ]

        try:
            response = self._invoke_claude(prompt, tools)

            # Handle tool call response
            if 'content' in response and len(response['content']) > 0:
                content_item = response['content'][0]

                if content_item['type'] == 'tool_use' and content_item['name'] == 'speak':
                    tool_input = content_item['input']
                    message = tool_input.get("message", "")

                    # Return the discussion contribution
                    discussion_entry = {
                        "agent_id": self.agent_id,
                        "message": message
                    }

                    self.logger.info(f"{self.name} ({self.agent_id}) contributed to the discussion:\n{message}")
                    return discussion_entry

                else:
                    raise ValueError("Invalid tool use")

        except Exception as e:
            # Log the error and return None if there's a problem
            self.logger.error(f"Error in LlmAgent {self.agent_id} discussion: {e}")
            return None

    def _invoke_claude(self, prompt: str, tools: List[Dict] = []) -> Dict:
        """Invoke Claude through AWS Bedrock with tool calling."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Anthropic Claude API format with tool calling
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "any"}

        # Exponential backoff parameters
        max_retries = 9 # max time delay is 256 seconds (2^8)
        retry_count = 0
        base_delay = 1  # Start with 1 second delay

        while True:
            try:
                response = self.bedrock.invoke_model(
                    modelId=bedrock_model_id,
                    body=json.dumps(payload)
                )
                response_body = json.loads(response['body'].read().decode('utf-8'))
                return response_body

            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')

                # Check if it's a throttling exception
                if error_code == 'ThrottlingException' and retry_count < max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    self.logger.warning(f"ThrottlingException encountered, retrying in {delay} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Either not a throttling exception or max retries exceeded
                    self.logger.error(f"Could not vote, Error invoking Claude: {e}")
                    raise

    def promptize_vote_history(self) -> str:
        if self.vote_history:
            vote_history_str = "\n".join([f"In round {i+1}, you voted for {self.env.agents[vote['vote']].name} for the following reason:\n{vote['reasoning']}" for i,vote in enumerate(self.vote_history)])
            return f"Your vote history:\n{vote_history_str}\n"
        else:
            return ""

    def promptize_voting_results_history(self) -> str:
        def promptize_voting_results(results: Dict[str, int]) -> str:
            voting_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            if results:
                voting_results_str = "\n".join([f"Cardinal {i} - {self.env.agents[i].name}: {votes}" for i, votes in voting_results])
                return f"\n{voting_results_str}\n"
            else:
                return ""

        if self.env.votingHistory:
            voting_results_history_str = "\n".join([f"Round {i+1}: {promptize_voting_results(result)}" for i,result in enumerate(self.env.votingHistory)])
            return f"Previous ballot results:\n{voting_results_history_str}"
        else:
            return ""
