from sentence_transformers import SentenceTransformer, util
import re

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []

    for prompt, completion in zip(prompts, completions):
        q = prompt[-1]["content"]  # user 질문
        response = completion[0]["content"]  # 모델 응답 텍스트
        
        # Ground-truth answer (kwargs에서 꺼냄)
        gt_answer = kwargs["answer"].pop(0).strip().lower()

        # Extract answer from response
        pred_answer = None
        
        # First try to find answer between <answer> tags
        answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", response, flags=re.IGNORECASE)
        if answer_match:
            pred_answer = answer_match.group(1).strip().lower()
            # Remove any remaining tags or special characters
            pred_answer = re.sub(r'<[^>]+>', '', pred_answer)
            # Take only the first word and remove any special characters
            pred_answer = re.sub(r'[^a-zA-Z0-9]', '', pred_answer.split()[0])
        
        # If no valid answer found, use a default value
        if not pred_answer:
            pred_answer = "invalid"

        # Calculate semantic similarity
        emb1 = semantic_model.encode(gt_answer, convert_to_tensor=True)
        emb2 = semantic_model.encode(pred_answer, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
            
        rewards.append(round(sim, 4))

        # 디버깅 출력
        print("-" * 40)
        print(f"Response: {response}")
        print(f"Q: {q}")
        print(f"GT : {gt_answer}")
        print(f"Pred : {pred_answer}")
        print(f"Semantic reward: {sim}")

    return rewards

def format_reward(completions, **kwargs) -> list[float]:
    """
    Evaluate if the response follows the required format:
    1. Has both <thought> and <answer> tags
    2. Answer is a single word
    3. No text outside the tags
    4. No other tags or formats used
    
    Returns a reward between 0.0 and 1.0
    """
    rewards = []
    
    for completion in completions:
        response = completion[0]["content"]
        
        # Check for required tags
        has_thought = bool(re.search(r"<thought>.*?</thought>", response, flags=re.IGNORECASE | re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", response, flags=re.IGNORECASE | re.DOTALL))
        
        # Extract answer to check if it's a single word
        answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", response, flags=re.IGNORECASE)
        is_single_word = False
        if answer_match:
            answer = answer_match.group(1).strip()
            # Remove any remaining tags
            answer = re.sub(r'<[^>]+>', '', answer)
            # Check if it's a single word (no spaces)
            is_single_word = len(answer.split()) == 1
        
        # Check for text outside tags
        has_text_outside = False
        if has_thought and has_answer:
            # Remove thought and answer tags
            text_without_tags = re.sub(r'<thought>.*?</thought>', '', response, flags=re.IGNORECASE | re.DOTALL)
            text_without_tags = re.sub(r'<answer>.*?</answer>', '', text_without_tags, flags=re.IGNORECASE | re.DOTALL)
            # Check if there's any non-whitespace text left
            has_text_outside = bool(re.search(r'\S', text_without_tags))
        
        # Check for other tags
        has_other_tags = bool(re.search(r'<(?!thought|answer)[^>]+>', response, flags=re.IGNORECASE))
        
        # Calculate reward
        reward = 1.0
        
        # Apply penalties
        if not has_thought or not has_answer:
            reward *= 0.3  # Heavy penalty for missing required tags
        if not is_single_word:
            reward *= 0.5  # Penalty for multi-word answers
        if has_text_outside:
            reward *= 0.7  # Penalty for text outside tags
        if has_other_tags:
            reward *= 0.7  # Penalty for using other tags
            
        rewards.append(round(reward, 4))
        
        # Debug output
        print("-" * 40)
        print(f"Response: {response}")
        print(f"Format check:")
        print(f"- Has thought tag: {has_thought}")
        print(f"- Has answer tag: {has_answer}")
        print(f"- Is single word: {is_single_word}")
        print(f"- Has text outside tags: {has_text_outside}")
        print(f"- Has other tags: {has_other_tags}")
        print(f"Format reward: {reward}")
        
    return rewards