from typing import List

from openai import OpenAI

from utils.get_env import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# TODO -> FOR BUILD THIS SYSTEM:
#  1 SOLUTION: NEED TO INSTALL OLD VERSION OF "OPENAI" LIBRARY == 0.28
#  2 SOLUTION: FIND ANOTHER SOLUTION WITH THIS VERSION "OPENAI" LIBRARY :)

def recommendations_from_strings(
        strings: List[str],
        index_of_source_string: int,
        model="text-embedding-3-small",
) -> List[int]:
    """Return nearest neighbors of a given string."""

    # get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]

    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]

    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    return indices_of_nearest_neighbors
