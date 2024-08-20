def wrap_text_over_words(title: str, max_line_length: int):
    """
    Wraps the title string into lines with a maximum specified length without breaking words.

    Parameters:
    title (str): The original title string.
    max_line_length (int): The maximum number of characters per line.

    Returns:
    str: The wrapped title string.
    """
    current_line_length = 0
    lines = []
    current_line = []

    words = title.split()

    for word in words:
        # Check if adding the word would exceed the max length
        if current_line_length + len(word) + (1 if current_line else 0) > max_line_length:
            # If it exceeds, append the current line to lines and reset
            lines.append(' '.join(current_line))
            current_line = []
            current_line_length = 0

        # Add the word to the current line
        current_line_length += len(word) + (1 if current_line else 0)  # Add 1 for the space if it's not the first word
        current_line.append(word)

    # Append the last line
    if current_line:
        lines.append(' '.join(current_line))

    # Join the lines with newlines
    return '\n'.join(lines)
