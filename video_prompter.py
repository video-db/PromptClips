import json

import concurrent.futures

from videodb import connect
from videodb import SearchType, IndexType
from videodb.timeline import VideoAsset


from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """
    Get connection and load the env.
    :return:
    """
    conn = connect()
    return conn


def get_video(id):
    """
    Get video object
    :param id:
    :return:
    """
    conn = get_connection()
    all_videos = conn.get_collection().get_videos()
    video = next(vid for vid in all_videos if vid.id == id)
    return video


def chunk_docs(docs, chunk_size):
    """
    chunk docs to fit into context of your LLM
    :param docs:
    :param chunk_size:
    :return:
    """
    for i in range(0, len(docs), chunk_size):
        yield docs[i : i + chunk_size]


def get_result_timestamps(
    video,
    result,
    index_type="scene",
    scene_index_id=None,
    sort="time",
    run_concurrent=True,
):
    """
    This function takes the result from scene_prompter and performs a keyword search on the video.
    By default, the function sorts the results by time.
    It returns a list of (start, end, description) for the matched segments.
    """
    result_timestamps = []

    def search_description(description):
        if index_type == "scene":
            search_res = video.search(
                description,
                index_type=IndexType.scene,
                search_type=SearchType.keyword,
                scene_index_id=scene_index_id,
            )
        else:
            search_res = video.search(
                description,
                index_type=IndexType.spoken_word,
                search_type=SearchType.keyword,
            )
        matched_segments = search_res.get_shots()
        if len(matched_segments) == 0:
            return None
        video_shot = matched_segments[0]
        return (video_shot.start, video_shot.end, video_shot.text)

    if run_concurrent:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_desc = {
                executor.submit(search_description, desc): desc for desc in result
            }
            for future in concurrent.futures.as_completed(future_to_desc):
                res = future.result()
                if res:
                    result_timestamps.append(res)
    else:
        for description in result:
            res = search_description(description)
            if res:
                result_timestamps.append(res)

    if sort == "time":
        result_timestamps.sort(key=lambda x: x[0])

    #Print the found segments
    print(f"\nFOUND SEGMENTS ({len(result_timestamps)}):")
    for start, end, description in result_timestamps:
        print(f"{start:.3f}-{end:.3f}: {description}")

    return result_timestamps


def build_video_timeline(
    video, result_timestamps, timeline, top_n=None, max_duration=None, debug=False
):
    duration = 0
    added_clips = 0
    added_segments = []

    if top_n:
        existing_count = len(result_timestamps)
        result_timestamps = result_timestamps[:top_n]
        if debug:
            print(f"Picked top {top_n} from {existing_count}")

    for result_timestamp in result_timestamps:
        start = float(result_timestamp[0])
        end = float(result_timestamp[1])
        description = result_timestamp[2]

        if debug:
            print(f"Adding clip {added_clips + 1}: {start} - {end}, Description: {description}")

        if max_duration and duration + (end - start) > max_duration:
            print("Max duration reached. Stopping further additions.")
            break

        timeline.add_inline(VideoAsset(asset_id=video.id, start=start, end=end))
        duration += end - start
        added_clips += 1
        added_segments.append((start, end, description))  #Collects added segments

    #Prints the added segments
    print(f"\nSEGMENTS ADDED ({len(added_segments)}):")
    for start, end, description in added_segments:
        print(f"{start:.3f}-{end:.3f}: {description}")

    return timeline, duration



def filter_transcript(transcript, start, end):
    result = []
    for entry in transcript:
        if float(entry["end"]) > start and float(entry["start"]) < end:
            result.append(entry)
    return result


def get_multimodal_docs(transcript, scenes, club_on="scene"):
    docs = []
    if club_on == "scene":
        for scene in scenes:
            spoken_result = filter_transcript(
                transcript, float(scene["start"]), float(scene["end"])
            )
            spoken_text = " ".join(
                entry["text"] for entry in spoken_result if entry["text"] != "-"
            )
            data = {
                "visual": scene["description"],
                "spoken": spoken_text,
                "start": scene["start"],
                "end": scene["end"],
            }
            docs.append(data)
    return docs


def send_msg_openai(chunk_prompt, llm=LLM()):
    response = llm.chat(message=chunk_prompt)
    output = json.loads(response["choices"][0]["message"]["content"])
    sentences = output.get("sentences")
    return sentences


def send_msg_claude(chunk_prompt, llm):
    response = llm.chat(message=chunk_prompt)
    return response


def send_msg_gemini(chunk_prompt, llm):
    response = llm.chat(message=chunk_prompt)
    return response


def text_prompter(transcript_text, prompt, llm=None):
    chunk_size = 10000
    chunks = chunk_docs(transcript_text, chunk_size=chunk_size)

    if llm is None:
        llm = LLM()

    if llm.type == LLMType.OPENAI:
        llm_caller_fn = send_msg_openai
    elif llm.type == LLMType.GEMINI:
        llm_caller_fn = send_msg_gemini
    else:
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0
    for chunk in chunks:
        chunk_prompt = """
        You are a video editor who uses AI. Given a user prompt and transcript of a video analyze the text to identify sentences in the transcript relevant to the user prompt for making clips.
        - **Instructions**:
          - Evaluate the sentences for relevance to the specified user prompt.
          - Make sure that sentences start and end properly and meaningfully complete the discussion or topic. Choose the one with the greatest relevance and longest.
          - We'll use the sentences to make video clips in future, so optimize for great viewing experience for people watching the clip of these.
          - If the matched sentences are not too far, merge them into one sentence.
          - Strictly make each result minimum 20 words long. If the match is smaller, adjust the boundries and add more context around the sentences.

        - **Output Format**: Return a JSON list of strings named 'sentences' that containes the output sentences, make sure they are exact substrings.
        - **User Prompts**: User prompts may include requests like 'find funny moments' or 'find moments for social media'. Interpret these prompts by
        identifying keywords or themes in the transcript that match the intent of the prompt.
        """

        # pass the data
        chunk_prompt += f"""
        Transcript: {chunk}
        User Prompt: {prompt}
        """

        # Add instructions to always return JSON at the end of processing.
        chunk_prompt += """
        Ensure the final output strictly adheres to the JSON format specified without including additional text or explanations. \
        If there is no match return empty list without additional text. Use the following structure for your response:
        {
          "sentences": [
            {},
            ...
          ]
        }
        """
        prompts.append(chunk_prompt)
        i += 1

    # make a parallel call to all chunks with prompts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(llm_caller_fn, prompt, llm): prompt for prompt in prompts
        }
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                matches.extend(future.result())
            except Exception as e:
                print(f"Chunk failed to work with LLM {str(e)}")
    return matches


def scene_prompter(transcript_text, prompt, llm=None, run_concurrent=True):
    chunk_size = 200
    chunks = chunk_docs(transcript_text, chunk_size=chunk_size)

    llm_caller_fn = send_msg_gemini
    if llm is None:
        llm = LLM()

    # TODO:  llm should have caller function
    # 400 sentence at a time
    if llm.type == LLMType.GEMINI:
        llm_caller_fn = send_msg_gemini
    else:
        # claude for now
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0

    for chunk in chunks:
        descriptions = [scene["description"] for scene in chunk]
        chunk_prompt = """
        You are a video editor who uses AI. Given a user prompt and AI-generated scene descriptions of a video, analyze the descriptions to identify segments relevant to the user prompt for creating clips.

        - **Instructions**:
            - Evaluate the scene descriptions for relevance to the specified user prompt.
            - Choose description with the highest relevance and most comprehensive content.
            - Optimize for engaging viewing experiences, considering visual appeal and narrative coherence.

            - User Prompts: Interpret prompts like 'find exciting moments' or 'identify key plot points' by matching keywords or themes in the scene descriptions to the intent of the prompt.
        """

        chunk_prompt += f"""
        Descriptions: {json.dumps(descriptions)}
        User Prompt: {prompt}
        """

        chunk_prompt += """
         **Output Format**: Return a JSON list of strings named 'result' that containes the  fileds `sentence` Ensure the final output
        strictly adheres to the JSON format specified without including additional text or explanations. \
        If there is no match return empty list without additional text. Use the following structure for your response:
        {"sentences": []}
        """
        prompts.append(chunk_prompt)
        i += 1

    if run_concurrent:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(llm_caller_fn, prompt, llm): prompt
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    matches.extend(future.result())
                except Exception as e:
                    print(f"Chunk failed to work with LLM {str(e)}")
    else:
        for prompt in prompts:
            try:
                res = llm_caller_fn(prompt, llm)
                matches.extend(res)
            except Exception as e:
                print(f"Chunk failed to work with LLM {str(e)}")
    return matches


def multimodal_prompter(transcript, scene_index, prompt, llm=None, run_concurrent=True):
    docs = get_multimodal_docs(transcript, scene_index)
    chunk_size = 80
    chunks = chunk_docs(docs, chunk_size=chunk_size)

    if llm is None:
        llm = LLM()

    if llm.type == LLMType.GEMINI:
        llm_caller_fn = send_msg_gemini
    else:
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0
    for chunk in chunks:
        chunk_prompt = f"""
        You are given visual and spoken information of the video of each second, and a transcipt of what's being spoken along with timestamp.
        Your task is to evaluate the data for relevance to the specified user prompt.
        Corelate visual and spoken content to find the relevant video segment.

        Multimodal Data:
        video: {chunk}
        User Prompt: {prompt}


        """
        chunk_prompt += """
         **Output Format**: Return a JSON list of strings named 'result' that containes the  fileds `sentence`.
        sentence is from the visual section of the input.
        Ensure the final output strictly adheres to the JSON format specified without including additional text or explanations.
        If there is no match return empty list without additional text. Use the following structure for your response:
        {"sentences": []}
        """
        prompts.append(chunk_prompt)
        i += 1

    if run_concurrent:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(llm_caller_fn, prompt, llm): prompt
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    matches.extend(future.result())
                except Exception as e:
                    print(f"Chunk failed to work with LLM {str(e)}")
    else:
        for prompt in prompts:
            try:
                res = llm_caller_fn(prompt)
                matches.extend(res)
            except Exception as e:
                import traceback

                print(traceback.print_exc())
                print(f"Chunk failed to work with LLM {str(e)}")
    return matches
