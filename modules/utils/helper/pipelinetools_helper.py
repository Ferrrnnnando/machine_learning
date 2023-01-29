# Others
from collections import Counter


def cumulatively_categorise(column, threshold=0.85, return_categories_list=True):
    """Group minority classes into a single class 'other' to reduce cardinality.

    Args:
        column (pd.DataFrame): Column to be aggregated.
        threshold (float, optional): Threshold for keeping class. Defaults to 0.85.
        return_categories_list (bool, optional): Return the transformed column. Defaults to True.

    Returns:
        pd.Series: transformed columns
    """
    # Find the threshold value using the percentage and number of instances in the column
    threshold_value = int(threshold * len(column))
    # Initialise an empty list for our new minimised categories
    categories_list = []
    # Initialise a variable to calculate the sum of frequencies
    s = 0
    # Create a counter dictionary of the form unique_value: frequency
    counts = Counter(column)

    # Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i, j in counts.most_common():
        # Add the frequency to the global sum
        s += dict(counts)[i]
        # Append the category name to the list
        categories_list.append(i)
        # Check if the global sum has reached the threshold value, if so break the loop
        if s >= threshold_value:
            break
    # Append the category Other to the list
    categories_list.append("Other")

    # Replace all instances not in our new categories by Other
    new_column = column.apply(lambda x: x if x in categories_list else "Other")

    # Return transformed column and unique values if return_categories=True
    if return_categories_list:
        return new_column, categories_list
    # Return only the transformed column if return_categories=False
    else:
        return new_column
