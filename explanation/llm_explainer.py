from transformers import pipeline
import torch
class LLMExplainer:
    def __init__(self, 
    # model_id="meta-llama/Llama-3.2-3B-Instruct"
    model
    ):
        """Initialize the LLM pipeline for text generation."""
        self.pipe = model

    def generate_explanation(self, user_preferences, item_features):
        """Generate a concise explanation for a recommended item."""
        # Format watch history entries
        watch_history_strings = [
            f"'{entry['title']}' ({'Liked' if entry['liked'] else 'Disliked'}): {', '.join(entry['genres'])}" 
            for entry in user_preferences["watch_history"]
        ]
        
        # Prepare system and user message content
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI system that generates highly relevant, extremely concise explanations for recommended items. "
                    "Follow these rules strictly:\n"
                    "1. Justification: Focus on matching genres, shared themes with watch history, or specific keywords aligned with user interests. "
                    "Only include information that directly supports the recommendation.\n"
                    "2. Concise Explanation: Provide exactly 1-2 short sentence (under 40 words if possible). Clearly justify why this item is recommended. "
                    "Avoid greetings, filler, or unnecessary elaboration.\n"
                    "3. Handling Disliked Genres: If the item belongs to a disliked genre, return: 'This is from a genre you typically avoid, but it might still interest you.'\n"
                    "4. Non-Matching Items: If no meaningful connection exists, return: 'This may not fully match your preferences but could offer a new experience.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Explain in one short sentence why '{item_features['title']}' is recommended based on my preferences:\n\n"
                    f"User Preferences:\n"
                    f"- Liked Genres: {', '.join(user_preferences['liked_genres'])}\n"
                    f"- Disliked Genres: {', '.join(user_preferences['disliked_genres'])}\n"
                    f"- Watch History: {', '.join(watch_history_strings)}\n\n"
                    f"Item Features:\n"
                    f"- Title: {item_features['title']}\n"
                    f"- Genre: {item_features['genre']}\n"
                    f"Ensure the explanation is extremely short, directly relevant, and avoids unnecessary details."
                )
            }
        ]
        
        # Get response
        response = self.pipe(messages, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=128001)
        
        # Return the generated text from the response
        return response[0]["generated_text"][-1]['content']
