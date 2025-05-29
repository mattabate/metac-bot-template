import asyncio
import datetime
import json
import os
import re
import dotenv

dotenv.load_dotenv()

from openai import AsyncOpenAI
import numpy as np
import requests
from joseph import async_call_research
from prompts import (
    BINARY_PROMPT_TEMPLATE,
    BINARY_PROMPT_TEMPLATE_RESEARCH,
    NUMERIC_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE_RESEARCH,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE_RESEARCH,
)

######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 3  # The median/average forecast is taken between NUM_RUNS_PER_QUESTION runs (each with independent research)
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True

# Environment variables
# You only need *either* Exa or Perplexity or AskNews keys for online research
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # You'll also need the OpenAI API Key if you want to use the Exa Smart Searcher

# The tournament IDs below can be used for testing your bot.
Q2_2025_AI_BENCHMARKING_ID = 32721
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
AXC_2025_TOURNAMENT_ID = 32564
GIVEWELL_ID = 3600
RESPIRATORY_OUTLOOK_ID = 3411

TOURNAMENT_ID = Q2_2025_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (
        578,
        578,
    ),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (
        14333,
        14333,
    ),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (
        22427,
        22427,
    ),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
]

# Also, we realize the below code could probably be cleaned up a bit in a few places
# Though we are assuming most people will dissect it enough to make this not matter much

######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    print(response.status_code, response.headers.get("X-RateLimit-Remaining"))
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def list_posts_from_general(offset: int = 0, count: int = 50) -> list[dict]:
    """
    List (all details) {count} posts from all tournaments (no tournament filter)
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        # Removed "statuses": "open" to see all questions
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    print(response.status_code, response.headers.get("X-RateLimit-Remaining"))
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()
    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"Found open question - ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    print(f"Found {len(open_question_id_post_id)} open questions")

    if len(open_question_id_post_id) == 0:
        print(
            "No open questions found in the current tournament. Trying other tournaments..."
        )
        # Try other tournaments if current one has no questions
        other_tournaments = [
            Q1_2025_AI_BENCHMARKING_ID,
            Q4_2024_AI_BENCHMARKING_ID,
            Q1_2025_QUARTERLY_CUP_ID,
        ]
        for tournament_id in other_tournaments:
            if tournament_id != TOURNAMENT_ID:
                print(f"Trying tournament {tournament_id}...")
                posts = list_posts_from_tournament(tournament_id)
                for post in posts["results"]:
                    if question := post.get("question"):
                        if question.get("status") == "open":
                            print(
                                f"Found in tournament {tournament_id} - ID: {question['id']}\nQ: {question['title']}\nCloses: "
                                f"{question['scheduled_close_time']}"
                            )
                            open_question_id_post_id.append(
                                (question["id"], post["id"])
                            )
                if len(open_question_id_post_id) > 0:
                    break

        # If still no questions, try fetching from general open questions (no tournament filter)
        if len(open_question_id_post_id) == 0:
            print(
                "No open questions found in any tournament. Trying general open questions..."
            )
            try:
                posts = list_posts_from_general()
                for post in posts["results"]:
                    if question := post.get("question"):
                        if question.get("status") == "open":
                            print(
                                f"Found general question - ID: {question['id']}\nQ: {question['title']}\nCloses: "
                                f"{question['scheduled_close_time']}"
                            )
                            open_question_id_post_id.append(
                                (question["id"], post["id"])
                            )
                            if (
                                len(open_question_id_post_id) >= 3
                            ):  # Limit to 3 questions for testing
                                break
                # If still no open questions, try any status for testing
                if len(open_question_id_post_id) == 0:
                    print(
                        "No open questions found, trying questions with any status..."
                    )
                    for post in posts["results"][:3]:  # Just first 3 for testing
                        if question := post.get("question"):
                            print(
                                f"Found question - ID: {question['id']}\nQ: {question['title']}\nStatus: {question.get('status')}"
                            )
                            open_question_id_post_id.append(
                                (question["id"], post["id"])
                            )
            except Exception as e:
                print(f"Error fetching general questions: {e}")

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok or response.status_code != 200:
        print("!! MATT DUMB")
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


async def call_llm(prompt: str, model: str = "gpt-4o", temperature: float = 0.3) -> str:
    """
    Makes a streaming completion request to OpenAI's API with concurrent request limiting.
    """

    # Remove the base_url parameter to call the OpenAI API directly
    # Also checkout the package 'litellm' for one function that can call any model from any provider
    # Email ben@metaculus.com if you need credit for the Metaculus OpenAI/Anthropic proxy

    print("\033[92m> prompt \033[0m")
    print("\033[93m" + prompt + "\033[0m")

    client = AsyncOpenAI(
        max_retries=2,
    )

    async with llm_rate_limiter:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("No answer returned from LLM")
        return answer


async def run_research(question: str, max_retries: int = 2) -> str:
    """
    Run research with timeout and retry mechanism.
    Will timeout after 35 minutes and retry up to max_retries times.
    """
    timeout_minutes = 35
    timeout_seconds = timeout_minutes * 60

    for attempt in range(max_retries + 1):
        try:
            print(
                f"Starting research attempt {attempt + 1}/{max_retries + 1} (timeout: {timeout_minutes} minutes)"
            )

            # Use asyncio.wait_for to add timeout
            research = await asyncio.wait_for(
                async_call_research(question), timeout=timeout_seconds
            )

            print(
                f"########################\nResearch Found:\n{research}\n########################"
            )

            return research

        except asyncio.TimeoutError:
            print(
                f"Research attempt {attempt + 1} timed out after {timeout_minutes} minutes"
            )

            if attempt < max_retries:
                print(f"Retrying research... (attempt {attempt + 2}/{max_retries + 1})")
                # Small delay before retry
                await asyncio.sleep(5)
            else:
                print(
                    "All research attempts failed due to timeout. Returning fallback message."
                )
                exit()
                return "Research timed out after multiple attempts. No additional research data available."

        except Exception as e:
            print(f"Research attempt {attempt + 1} failed with error: {e}")

            if attempt < max_retries:
                print(f"Retrying research... (attempt {attempt + 2}/{max_retries + 1})")
                await asyncio.sleep(5)
            else:
                print("All research attempts failed. Returning fallback message.")
                exit()
                return f"Research failed after multiple attempts. Error: {str(e)}"


############### BINARY ###############
# @title Binary prompt & functions

# This section includes functionality for binary questions.


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_single_binary_prediction(
    question_details: dict, run_number: int
) -> tuple[float, str]:
    """
    Get a single binary prediction with its own research.
    """
    print(f"Starting binary prediction run {run_number}...")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    # Do research for this specific prediction (with independent timeout/retry)
    try:
        summary_report = await run_research(
            BINARY_PROMPT_TEMPLATE_RESEARCH.format(
                title=title,
                today=today,
                background=background,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
            )
        )
        print(f"Research completed for binary prediction run {run_number}")
    except Exception as e:
        print(f"Research failed for binary prediction run {run_number}: {e}")
        summary_report = f"Research failed for this run: {str(e)}"

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    rationale = await call_llm(content)
    probability = extract_probability_from_response_as_percentage_not_decimal(rationale)

    comment = (
        f"Run {run_number} - Extracted Probability: {probability}%\n\nGPT's Answer: "
        f"{rationale}\n\n\n"
    )
    print(f"Completed binary prediction run {run_number}: {probability}%")
    return probability, comment


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[float, str]:
    """
    Get binary prediction by running research + prediction num_runs times in parallel and taking median.
    """
    print(f"Running {num_runs} parallel research + prediction cycles...")

    # Run independent research + prediction cycles in parallel
    prediction_tasks = [
        get_single_binary_prediction(question_details, i + 1) for i in range(num_runs)
    ]
    probability_and_comment_pairs = await asyncio.gather(
        *prediction_tasks, return_exceptions=True
    )

    # Handle any exceptions and filter successful results
    successful_results = []
    comments = []

    for i, result in enumerate(probability_and_comment_pairs):
        if isinstance(result, Exception):
            print(f"Binary prediction run {i+1} failed: {result}")
            comments.append(f"Run {i+1} failed: {str(result)}")
        else:
            successful_results.append(result)
            comments.append(result[1])

    if not successful_results:
        raise RuntimeError("All binary prediction runs failed")

    final_comment_sections = [
        f"## Research + Prediction Run {i+1}\n{comment}"
        for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in successful_results]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = (
        f"Median Probability: {median_probability} (from {len(successful_results)}/{num_runs} successful runs)\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return median_probability, final_comment


####################### NUMERIC ###############
# @title Numeric prompt & functions


def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = (
            r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        )
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num) for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust any values that are exactly at the bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value

    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }

    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                list_index_2 = i

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i - 1], known_x[i]
                    y0, y1 = known_y[i - 1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)
    return continuous_cdf


async def get_single_numeric_prediction(
    question_details: dict, run_number: int
) -> tuple[list[float], str]:
    """
    Get a single numeric prediction with its own research.
    """
    print(f"Starting numeric prediction run {run_number}...")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = (
        question_details["unit"]
        if question_details["unit"]
        else "Not stated (please infer this)"
    )
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]

    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    # Do research for this specific prediction (with independent timeout/retry)
    try:
        summary_report = await run_research(
            NUMERIC_PROMPT_TEMPLATE_RESEARCH.format(
                title=title,
                today=today,
                background=background,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                lower_bound_message=lower_bound_message,
                upper_bound_message=upper_bound_message,
                units=unit_of_measure,
            )
        )
        print(f"Research completed for numeric prediction run {run_number}")
    except Exception as e:
        print(f"Research failed for numeric prediction run {run_number}: {e}")
        summary_report = f"Research failed for this run: {str(e)}"

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        units=unit_of_measure,
    )

    rationale = await call_llm(content)
    percentile_values = extract_percentiles_from_response(rationale)

    comment = (
        f"Run {run_number} - Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: "
        f"{rationale}\n\n\n"
    )

    cdf = generate_continuous_cdf(
        percentile_values,
        question_type,
        open_upper_bound,
        open_lower_bound,
        upper_bound,
        lower_bound,
        zero_point,
    )

    print(f"Completed numeric prediction run {run_number}")
    return cdf, comment


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[list[float], str]:
    """
    Get numeric prediction by running research + prediction num_runs times in parallel and taking median.
    """
    print(f"Running {num_runs} parallel research + prediction cycles...")

    # Run independent research + prediction cycles in parallel
    prediction_tasks = [
        get_single_numeric_prediction(question_details, i + 1) for i in range(num_runs)
    ]
    cdf_and_comment_pairs = await asyncio.gather(
        *prediction_tasks, return_exceptions=True
    )

    # Handle any exceptions and filter successful results
    successful_results = []
    comments = []

    for i, result in enumerate(cdf_and_comment_pairs):
        if isinstance(result, Exception):
            print(f"Numeric prediction run {i+1} failed: {result}")
            comments.append(f"Run {i+1} failed: {str(result)}")
        else:
            successful_results.append(result)
            comments.append(result[1])

    if not successful_results:
        raise RuntimeError("All numeric prediction runs failed")

    final_comment_sections = [
        f"## Research + Prediction Run {i+1}\n{comment}"
        for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in successful_results]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment = (
        f"Median CDF: `{str(median_cdf)[:100]}...` (from {len(successful_results)}/{num_runs} successful runs)\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return median_cdf, final_comment


########################## MULTIPLE CHOICE ###############
# @title Multiple Choice prompt & functions


def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_single_multiple_choice_prediction(
    question_details: dict, run_number: int
) -> tuple[dict[str, float], str]:
    """
    Get a single multiple choice prediction with its own research.
    """
    print(f"Starting multiple choice prediction run {run_number}...")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    # Do research for this specific prediction (with independent timeout/retry)
    try:
        summary_report = await run_research(
            MULTIPLE_CHOICE_PROMPT_TEMPLATE_RESEARCH.format(
                title=title,
                today=today,
                background=background,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                options=options,
            )
        )
        print(f"Research completed for multiple choice prediction run {run_number}")
    except Exception as e:
        print(f"Research failed for multiple choice prediction run {run_number}: {e}")
        summary_report = f"Research failed for this run: {str(e)}"

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    rationale = await call_llm(content)
    option_probabilities = extract_option_probabilities_from_response(
        rationale, options
    )

    comment = (
        f"Run {run_number} - EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
        f"{rationale}\n\n\n"
    )

    probability_yes_per_category = generate_multiple_choice_forecast(
        options, option_probabilities
    )
    print(f"Completed multiple choice prediction run {run_number}")
    return probability_yes_per_category, comment


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str]:
    """
    Get multiple choice prediction by running research + prediction num_runs times in parallel and taking average.
    """
    print(f"Running {num_runs} parallel research + prediction cycles...")

    options = question_details["options"]

    # Run independent research + prediction cycles in parallel
    prediction_tasks = [
        get_single_multiple_choice_prediction(question_details, i + 1)
        for i in range(num_runs)
    ]
    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *prediction_tasks, return_exceptions=True
    )

    # Handle any exceptions and filter successful results
    successful_results = []
    comments = []

    for i, result in enumerate(probability_yes_per_category_and_comment_pairs):
        if isinstance(result, Exception):
            print(f"Multiple choice prediction run {i+1} failed: {result}")
            comments.append(f"Run {i+1} failed: {str(result)}")
        else:
            successful_results.append(result)
            comments.append(result[1])

    if not successful_results:
        raise RuntimeError("All multiple choice prediction runs failed")

    final_comment_sections = [
        f"## Research + Prediction Run {i+1}\n{comment}"
        for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in successful_results
    ]

    # Calculate average probabilities across all successful runs
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}` (from {len(successful_results)}/{num_runs} successful runs)\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment


################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast
    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(
        f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
    )
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    # return summary_of_forecast
    if submit_prediction == True:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    print("\n", "#" * 100, "\nProcessing Questions Sequentially\n", "#" * 100)

    forecast_summaries = []
    errors = []

    for i, (question_id, post_id) in enumerate(open_question_id_post_id, 1):
        print(f"\n--- Processing Question {i}/{len(open_question_id_post_id)} ---")
        try:
            forecast_summary = await forecast_individual_question(
                question_id,
                post_id,
                submit_prediction,
                num_runs_per_question,
                skip_previously_forecasted_questions,
            )
            forecast_summaries.append(forecast_summary)
            print(forecast_summary)
        except Exception as e:
            error_msg = f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {e.__class__.__name__} {e}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            print(error_msg)
            errors.append(e)
            forecast_summaries.append(error_msg)

    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)
    for summary in forecast_summaries:
        print(summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)


######################## FINAL RUN #########################
if __name__ == "__main__":
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
