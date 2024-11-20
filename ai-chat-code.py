from typing import List, Dict, Any
from langgraph.graph import Graph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
import os
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Set AWS credentials (optional if set in environment variables)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Initialize the Anthropic Claude model via Amazon Bedrock
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Replace with your model ID
    region_name="ap-northeast-1",
    temperature=0.4,
    max_tokens = 16000,
)

# Define the state type
StateType = Dict[str, Any]

# Define tools (nodes) as functions
def job_description_analysis(state: StateType) -> StateType:
    prompt_template = PromptTemplate.from_template(
        "Analyze this job description and extract key responsibilities and skills: {job_description}"
    )
    prompt = prompt_template.format(job_description=state["job_description"])
    result = llm.invoke(prompt).content
    print(result)
    state["key_responsibilities"] = result
    return state

class Subtopic(BaseModel):
    name: str = Field(description="Name of the subtopic")
    priority: str = Field(description="Priority level of the subtopic (high/medium/low)")

class BroaderTopic(BaseModel):
    broaderTopic: str = Field(description="Name of the broader topic category")
    subtopics: List[Subtopic] = Field(description="List of subtopics under this broader topic")

class TopicSet(BaseModel):
    broaderTopics: List[BroaderTopic] = Field(description="List of all broader topics with their subtopics")

# Create a parser
parser_topic_gen = PydanticOutputParser(pydantic_object=TopicSet)

def topic_generation(state: StateType) -> StateType:
    prompt_template = PromptTemplate.from_template(
        '''You are an expert in analyzing job descriptions and identifying key topics and subtopics that would be important for assessment.
        Given the following job description, identify exactly {num_broader_topics} broader topics and exactly {num_subtopics} subtopics per broader topic that should be assessed during evaluation.

        Job Description:
        {job_description}

        Guidelines:
        1. Generate exactly {num_broader_topics} broader topics
        2. For each broader topic, identify exactly {num_subtopics} subtopics
        3. Assign priority levels (high/medium/low) based on importance in the job description
        4. Include both technical and soft skills where applicable
        5. Consider both explicit and implicit skill requirements
        6. Ensure topics are distinct and non-overlapping
        7. Cover the most important aspects first

        {format_instructions}

        ### Important Notes:
        1. Respond **only in the JSON format**.
        2. Do not include additional text, comments, or explanations.
        3. Ensure the JSON is well-formed and adheres strictly to the schema provided.

        Ensure the output:
        - Contains exactly {num_broader_topics} broader topics
        - Each broader topic has exactly {num_subtopics} subtopics
        - Follows the exact schema provided
        - Has properly assigned priorities based on job requirements
        ''',
        partial_variables= {"format_instructions": parser_topic_gen.get_format_instructions()}
    )
    num_broader_topics=2
    num_subtopics=20
    prompt = prompt_template.format(job_description=state["key_responsibilities"], num_broader_topics=num_broader_topics, num_subtopics=num_subtopics)

    result = llm.invoke(prompt).content
    print(result)

    # print("DEBUG - Raw LLM Response:", result)  # Debugging
    
    # Parse the response into a Pydantic model
    try:
        parsed_response = parser_topic_gen.parse(result)
    except Exception as e:
        raise ValueError(f"Failed to parse response: {e}\nResponse: {result}")

    # Update the state with parsed topics
    state["topics"] = parsed_response.model_dump_json(indent=2)  # Correctly dump the JSON from the Pydantic model
    return state
    

class DifficultyCategories(BaseModel):
    veryHard: List[str] = Field(description="List of topics that are very difficult to master")
    hard: List[str] = Field(description="List of topics that are hard but not extremely difficult")
    medium: List[str] = Field(description="List of topics of moderate difficulty")
    easy: List[str] = Field(description="List of topics that are relatively easier to grasp")


parser_categorize = PydanticOutputParser(pydantic_object=DifficultyCategories)

def topic_categorization(state: StateType) -> StateType:
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert in categorizing technical topics by their difficulty level.
        Given the following list of broader topics and their subtopics from a job description, categorize ONLY the broader topics into difficulty levels.

        Input Topics:
        {topics}

        Guidelines for categorization:

        Very Hard:
        - Topics requiring deep theoretical knowledge and extensive practical experience
        - Topics involving complex mathematical or engineering principles
        - Topics requiring integration of multiple complex technical domains

        Hard:
        - Topics requiring significant technical expertise
        - Topics involving detailed understanding of processes and systems
        - Topics requiring several years of experience to master

        Medium:
        - Topics requiring moderate technical knowledge
        - Topics that can be learned through standard industry experience
        - Topics involving standard tools and methodologies

        Easy:
        - Topics that can be learned through basic training
        - Topics focused on general skills or standard procedures
        - Topics involving common industry practices or soft skills

        Analyze each broader topic considering:
        1. The complexity of its subtopics
        2. The required depth of knowledge
        3. The learning curve involved
        4. The interdependencies with other topics

        {format_instructions}

        Important:
        - Only output the json as in the specified format and no other information
        - Categorize ONLY the broader topics, not the subtopics
        - Each topic should appear in exactly one difficulty category
        - Consider the overall complexity of each topic, not just individual subtopics
        - Base the classification on industry standards and typical learning curves
        """,
        partial_variables= {"format_instructions": parser_categorize.get_format_instructions()}
    )
    prompt= prompt_template.format(topics=state["topics"])
    result = llm.invoke(prompt).content
    print(result)
    parsed_response = parser_categorize.parse(result)
    state["categorized_topics"] = parsed_response.model_dump_json(indent=2)
    return state


def question_style_diversification(state: StateType) -> StateType:
    prompt_template = PromptTemplate.from_template(
        '''# Chemical Engineering Assessment Style Generator 

        ## Input Format 

        ### Required Fields: 

        1. **Job Description (JD):** : {JD}
        2. **Broader Topic:**  {BT} [just pick one or few]
        3. **Sub-Topics:** : []

        ## Output Format - Only JSON and no other information.
        - Sample Output format: 
        
        {{
        "question_styles": [
            {{
            "style_name": "[Explicit and specific name of the assessment style]",
            "definition": "[Clear description of what this style entails]",
            "example": "[Concrete example question in this style]",
            "assessment_goal": "[Specific skills or knowledge being evaluated]",
            "suitable_for_topics": ["Array of relevant sub-topics from input"]
            }}
        ]
        }}

        ## Style Naming Conventions: 
        - Use explicit, descriptive names 
        - Include the primary assessment method in the name 
        - Format: [Assessment Type] - [Focus Area] 

        ## Evaluation Guidelines: 
        1. **Style Names Should:** 
            - Be self-explanatory and specific 
            - Indicate both method and content area 
            - Reflect the complexity level 
            - Align with job requirements 
        2. **Definitions Should:** 
            - Clearly state what the style involves 
            - Indicate the type of response expected 
            - Specify any special conditions or requirements 
            - Be concise yet comprehensive 
        3. **Examples Should:** 
            - Be directly related to the style 
            - Be specific enough to demonstrate the style 
            - Be realistic and industry-relevant 
            - Match the job level requirements 
        4. **Assessment Goals Should:** 
            - Specify clear evaluation criteria 
            - Link to job requirements 
            - Cover both technical and soft skills where relevant 
            - Be measurable or observable 
        5. **Topic Matching Should:** 
            - Only include relevant sub-topics from input 
            - Be specific rather than general 
            - Consider prerequisite knowledge 
            - Align with job requirements 

        ### Important Notes:
        1. Respond **only in the JSON format**.
        2. Do not include additional text, comments, or explanations.
        3. Ensure the JSON is well-formed and adheres strictly to the schema provided.
        
        ## Usage Notes: 
        1. Generate at least one style for each major job requirement and there should be a total 15 different styles generated.
        2. Ensure styles cover both technical and practical aspects 
        3. Match complexity to job level 
        4. Include styles that assess both specific knowledge and broader capabilities 
        5. Consider company/industry context when creating examples
'''
    )
    if "topics" not in state or "job_description" not in state:
        raise ValueError("Missing 'topics' or 'job_description' in state")

    # Format the prompt
    prompt = prompt_template.format(
        BT=state["topics"],
        JD=state["job_description"]
    )
    result = llm.invoke(prompt).content
    print(result)
    state["diversified_questions"] = result
    return state

import json

def collect_style_feedback(state: StateType) -> StateType:
    """
    Presents each generated question style to the user for feedback and updates the state with liked styles.
    """
    diversified_questions = state.get("diversified_questions")
    if not diversified_questions:
        raise ValueError("No 'diversified_questions' found in the state.")

    try:
        styles_json = json.loads(diversified_questions)
        question_styles = styles_json.get("question_styles", [])
    except json.JSONDecodeError:
        raise ValueError("'diversified_questions' is not valid JSON.")

    liked_styles = []
    disliked_styles = []

    print("\n### Please provide feedback on the generated question styles ###\n")

    for idx, style in enumerate(question_styles, start=1):
        print(f"Style {idx}:")
        print(f"  - **Name:** {style['style_name']}")
        print(f"  - **Definition:** {style['definition']}")
        print(f"  - **Example:** {style['example']}")
        print(f"  - **Assessment Goal:** {style['assessment_goal']}")
        print(f"  - **Suitable Topics:** {', '.join(style['suitable_for_topics'])}\n")

        while True:
            feedback = input("Do you **like** this style? (enter 'like' or 'dislike'): ").strip().lower()
            if feedback in ['like', 'dislike']:
                break
            else:
                print("Invalid input. Please enter 'like' or 'dislike'.")

        if feedback == 'like':
            liked_styles.append(style)
        else:
            disliked_styles.append(style)
        print()  # Add a newline for better readability

    if not liked_styles:
        raise ValueError("No styles were liked. Please ensure at least one style is liked.")

    # Update the state with liked and disliked styles
    state["liked_question_styles"] = liked_styles
    state["disliked_question_styles"] = disliked_styles

    print("### Feedback Collection Complete ###\n")
    print(f"Liked Styles: {len(liked_styles)}")
    print(f"Disliked Styles: {len(disliked_styles)}\n")

    return state

def interlinking_question_creation(state: StateType) -> StateType:
    prompt_template = PromptTemplate.from_template(
        '''## Input Format
            ```json
            {{
            "jobDescription": {JD},
            "broaderTopic and subtopics": {broader_topic}
            }}
            ```
            ## Task Instructions
            1. Analyze the provided subtopics and job description
            2. Create logical pairs of topics that:
            - Demonstrate practical knowledge application
            - Test multiple competencies simultaneously
            - Reflect real-world problem-solving scenarios
            - Align with job responsibilities
            3. For each pair, provide:
            - Array of two or more related topics
            - Rationale for pairing
            - Brief example of assessment scenario
            - Relevance to job responsibilities


             ## Output Format - Only JSON and no other information.
            - Sample output format:

            {{
            "topicPairs": [{{
            "topics": "",
            "rationale": "",
            "assessmentExample": "",
            "jobRelevance": "",
            "priority": "high/medium/low"
            }}]
            }}
            

            ## Example Valid Response Excerpt
            
            {{
            "topicPairs": [
            {{
            "topics": ["Reactor Design", "Heat Exchanger Design"],
            "rationale": "Tests understanding of thermal management in reaction systems",
            "assessmentExample": "Design cooling system for exothermic batch reactor including heat exchanger specifications",
            "jobRelevance": "Directly relates to responsibilities #2, #4, and #6",
            "priority": "high"
            }},
            {{
            "topics": ["Process Flow Diagrams", "Material Balance"],
            "rationale": "Tests ability to develop and analyze complete process systems",
            "assessmentExample": "Develop PFD and material balance for a multi-step reaction process",
            "jobRelevance": "Addresses responsibilities #1, #2, and #10",
            "priority": "high"
            }}
            ]
            }}
            
            ### Important Notes:
            1. Respond **only in the JSON format**.
            2. Do not include additional text, comments, or explanations.
            3. Ensure the JSON is well-formed and adheres strictly to the schema provided.

            ## Constraints
            - Maximum 10 topic pairs to maintain focus
            - Each topic should appear in at least one combination
            - High-priority topics should appear in multiple combinations
            - Safety-related topics must be included in at least one combination

            ## Expected Topic Coverage
            The output should ensure:
            1. All high-priority topics appear in multiple combinations
            2. Each medium-priority topic appears at least once
            3. Low-priority topics are included where relevant to job responsibilities
            4. Safety considerations are integrated into appropriate combinations
        '''
    )
    prompt = prompt_template.format(JD = state["job_description"], broader_topic = state["topics"])
    result = llm.invoke(prompt).content
    print(result)
    state["interlinking_questions"] = result
    return state

def assessment_compilation(state: StateType) -> StateType:
    """
    Generates the final assessment questions based on liked question styles and interlinking topic pairs.
    """
    prompt_template = PromptTemplate.from_template(
        '''## Purpose
        Purpose: Generate specific assessment Multiple Choice questions based on chosen question styles and topic combinations. 
        
        ## Input Parameters
        1. **Liked Question Styles:** {liked_question_styles}
        2. **Topic Combinations:** {interlinking_response}
        3. **Job Description:** {JD}
        
        ## Instructions
        1. Using the selected question styles and topic combinations, generate a set of 10 questions that:
            - Follow each question style's approach.
            - Integrate both topics from the combination.
            - Align with the job level.
            - Reflect the industry context.
            - Map to specific job responsibilities.
        
        ## Output Format - Only JSON and no other information.
        - Sample Output format:
        {{
            "questions": [
                {{
                    "question": "[question without options catering to the instructions and input provided]",
                    "options": "[options for the question]",
                    "correct_answer": "[correct answer for the question]",
                    "style": "[style_name]",
                    "topics": [<topics>]
                }},
                ...
            ]
        }}
        
        
        ## Usage Guidelines
        1. Questions should require integration of both topics.
        2. Maintain consistency with the selected question styles.
        3. Include relevant industry context.
        4. Match complexity to job level.
        5. Align with rationale provided for topic combinations.
        
        ## Validation Criteria
        1. Question reflects the style definition.
        2. Both topics are meaningfully incorporated.
        3. Complexity matches job requirements.
        4. Context matches industry setting.
        5. Clear connection to job responsibilities.
        '''
    )

    # Ensure 'liked_question_styles' exists in the state
    if "liked_question_styles" not in state:
        raise ValueError("Missing 'liked_question_styles' in state.")

    # Serialize liked_question_styles to JSON string for the prompt
    liked_styles_json = json.dumps(state["liked_question_styles"], indent=2)

    prompt = prompt_template.format(
        liked_question_styles=liked_styles_json,
        JD=state["job_description"],
        interlinking_response=state["interlinking_questions"]
    )
    
    result = llm.invoke(prompt).content
    print("\n### Final Assessment Compilation Generated ###\n")
    print(result)
    
    state["final_assessment"] = result
    return state


# Create the graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("job_description", job_description_analysis)
workflow.add_node("topic_generation", topic_generation)
workflow.add_node("topic_categorization", topic_categorization)
workflow.add_node("question_style_diversification", question_style_diversification)
workflow.add_node("collect_style_feedback", collect_style_feedback)  # New Node
workflow.add_node("interlinking_question_creation", interlinking_question_creation)
workflow.add_node("assessment_compilation", assessment_compilation)

# Add edges to the graph
workflow.add_edge("job_description", "topic_generation")
workflow.add_edge("topic_generation", "topic_categorization")
workflow.add_edge("topic_categorization", "question_style_diversification")
workflow.add_edge("question_style_diversification", "collect_style_feedback")  # New Edge
workflow.add_edge("collect_style_feedback", "interlinking_question_creation")  # New Edge
workflow.add_edge("interlinking_question_creation", "assessment_compilation")

# Set the entry point
workflow.set_entry_point("job_description")

# Compile the graph
app = workflow.compile()

# Run the workflow
job_description = """
Job details-Chemical engineer
Acharya Group
Area: Ambernath, Maharashtra, India
Experience: 4-8 Years
Role: Manager-Design Engineering/Process Engineering
Industry type: Specialty Chemicals, Pharmaceuticals, agro-chemicals, intermediates plants
Employment: Fulltime/Contract
Key Responsibilities
Collecting plant data and providing design feedback to team
Performing material balance, energy balance for the plant along with utility calculations.
Determining sizes and specifications for equipment and instruments before procurement.
Designing of equipment like agitator, condenser & heat exchanger, pumping system, batch reactor, distillation columns.
Ability to setup or evaluate processes in product development lab (for studying or optimizing processes in small scale level before scaling the process upto plant level)
Providing solutions for optimization/improvement of the process and energy consumptions
Support operation team in improving plant performance in various fields (safety/environment, quality, capacity, costs)
Performance testing and start up support at various plant sites
Trouble shooting, debottlenecking experience in unit operations such as distillation, heat transfer, filtration, adsorption columns, autoclaves, gas sparged reactors etc.
Preparation of process flow diagrams (PFD) development.
Experience in running or understanding simulation software such as Aspen
Basic knowledge in running codes in Matlab/python
Should be an expert in using Excel
Proficient written and spoken English and strong communication skills
Experience and knowledge to effectively communicate issues and ideas
Role: Manager-Design Engineering/Process Engineering
Industry type: Specialty Chemicals, Pharmaceuticals, agro chemicals, pigments, intermediates plants
Employment: Fulltime/Contract
Education: B.Tech/B.E./M.Tech/M.E./M.S. in Chemical Engineering with experience of working in organic chemical manufacturing companies (such as bulk drugs, agro chemicals, specialty, pigments, intermediates plants)
"""

initial_state = {"job_description": job_description}
final_state = app.invoke(initial_state)

print(final_state)