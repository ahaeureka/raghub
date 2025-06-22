def get_openai_key_from_auth(auth: str) -> str:
    """
    Extracts the OpenAI API key from the provided authentication string.

    Args:
        auth (str): The authentication string, typically in the format
        "Bearer <API_KEY>".
    Returns:
        str: The extracted OpenAI API key.
    """
    if not auth.startswith("Bearer "):
        raise ValueError("Invalid authentication format. Expected 'Bearer <API_KEY>'.")
    return auth[len("Bearer ") :].strip()
